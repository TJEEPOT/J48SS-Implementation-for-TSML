/*
 * This file is part of the UEA Time Series Machine Learning (TSML) toolbox.
 *
 * The UEA TSML toolbox is free software: you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * The UEA TSML toolbox is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with the UEA TSML toolbox. If not, see <https://www.gnu.org/licenses/>.
 */

package weka.classifiers.trees;

import tsml.data_containers.TimeSeriesInstances;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Sourcable;
import weka.classifiers.trees.j48SS.*;
import weka.classifiers.trees.j48SS.jmetal.PseudoRandom;
import weka.core.*;
import weka.core.Capabilities.Capability;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;

import java.util.*;

/**
 * <!-- globalinfo-start -->
 * Class for building a decision tree classifier based on J48. This class can handle binary split for nominal,
 * numeric, time series (string) and continuous (string) attributes.<br>
 * <br>
 * For more information, see:<br>
 * <br>
 * Brunello et al. (2019). J48SS: A novel decision tree approach for the handling of sequential and time series data.
 * <br>
 * <p>
 * <!-- globalinfo-end -->
 *
 * <!-- technical-bibtex-start -->
 * BibTeX:
 * <pre>
 * &#64;article{brunello2019j48ss,
 *    title = {J48SS: A novel decision tree approach for the handling of sequential and time series data},
 *    author = {Brunello, Andrea and Marzano, Enrico and Montanari, Angelo and Sciavicco, Guido},
 *    journal = {Computers},
 *    volume = {8},
 *    number = {1},
 *    pages = {21},
 *    year = {2019},
 *    url = {https://doi.org/10.3390/computers8010021},
 *    publisher = {Multidisciplinary Digital Publishing Institute}
 * }
 *
 * &#64;book{Quinlan1993,
 *    address = {San Mateo, CA},
 *    author = {Ross Quinlan},
 *    publisher = {Morgan Kaufmann Publishers},
 *    title = {C4.5: Programs for Machine Learning},
 *    year = {1993}
 * }
 * </pre>
 * <p>
 * <!-- technical-bibtex-end -->
 *
 * <!-- options-start -->
 * Valid options are: </p>
 *
 * <pre> -U
 *  Use unpruned tree.</pre>
 *
 * <pre> -O
 *  Do not collapse tree.</pre>
 *
 * <pre> -C &lt;pruning confidence&gt;
 *  Set confidence threshold for pruning.
 *  (default 0.25)</pre>
 *
 * <pre> -M &lt;minimum number of instances&gt;
 *  Set minimum number of instances per leaf.
 *  (default 2)</pre>
 *
 * <pre> -R
 *  Use reduced error pruning.</pre>
 *
 * <pre> -N &lt;number of folds&gt;
 *  Set number of folds for reduced error
 *  pruning. One fold is used as pruning set.
 *  (default 3)</pre>
 *
 * <pre> -B
 *  Use binary splits only.</pre>
 *
 * <pre> -S
 *  Don't perform subtree raising.</pre>
 *
 * <pre> -L
 *  Do not clean up after the tree has been built.</pre>
 *
 * <pre> -A
 *  Laplace smoothing for predicted probabilities.</pre>
 *
 * <pre> -J
 *  Do not use MDL correction for info gain on numeric attributes.</pre>
 *
 * <pre> -Q &lt;seed&gt;
 *  Seed for random data shuffling (default 1).</pre>
 *
 * <!-- options-end -->
 *
 * @author Eibe Frank (eibe@cs.waikato.ac.nz) (J48 implementation)
 * @author Andrea Brunello (andrea.brunello{[at]}uniud{[dot]}it) (original J48SS weka implementation)
 * @author Martin Siddons (tjeepot{[at]}gmail{[dot]}com) (J48SS TSML implementation)
 * @version $Revision: 9117 $
 */
public class J48SS extends AbstractClassifier implements OptionHandler, Drawable, Matchable, Sourcable,
        WeightedInstancesHandler, Summarizable, AdditionalMeasureProducer, TechnicalInformationHandler,
        PartitionGenerator {
    /** for serialization */
    static final long serialVersionUID = 237314235325749481L;
    
    /** Initial tree of the classifier */
    protected ClassifierTree m_root;
    
    /** Should the tree remain unpruned? */
    protected boolean m_unpruned = false;
    
    /** True if the tree is to be collapsed. */
    protected boolean m_collapseTree = true;
    
    /** The confidence factor for pruning. */
    protected float m_CF = 0.25F;
    
    /** Minimum number of instances that have to occur in at least two subsets induced by split. */
    protected int m_minNumObj = 2;
    
    /** Use MDL correction when finding splits on numeric attributes? */
    protected boolean m_useMDLcorrection = true;
    
    /** If true, uses IG scoring of the best non-sequential attribute to guide the pruning of pattern search space */
    protected boolean m_useIGPruning = false;
    
    /** True if Laplace correction should be performed */
    protected boolean m_useLaplace = false;
    
    /** Should reduced error pruning be used? */
    protected boolean m_reducedErrorPruning = false;
    
    /** Number of folds for reduced error pruning. */
    protected int m_numFolds = 3;
    
    /** Binary splits on nominal attributes? */
    protected boolean m_binarySplits = false;
    
    /** Is subtree raising to be performed? */
    protected boolean m_subtreeRaising = true;
    
    /** False if cleanup should be performed after the tree has been built */
    protected boolean m_noCleanup = false;
    
    /** The minimum support that the extracted sequential patterns must have (between 0 and 1) */
    protected double m_minimumSupport = 0.5D;
    
    /** The maximum gap between two itemsets in a pattern (1 = No gap) */
    protected int m_maxGap = 2;
    
    /** The maximum pattern length in terms of itemset count */
    protected int m_maxPatternLength = 20;
    
    /** The weight used to evaluate the patterns */
    protected double m_patternWeight = 0.75D;
    
    /** The population size of the genetic algorithm i.e., the number of individuals */
    protected int m_popSize = 100;
    
    /**
     * The number of evaluations that are going to be carried out in the optimization process (should be higher than
     * popSize)
     */
    protected int m_numEvals = 500;
    
    /** The seed used by the genetic algorithm */
    protected int m_Seed = 1;
    
    /** The probability of combining two individuals of the population */
    protected double m_crossoverP = 0.8D;
    
    /** The probability of an element to undergo a random mutation */
    protected double m_mutationP = 0.1D;
    
    protected boolean m_doNotMakeSplitPointActualValue = false;
    
    /** Should the classifier print out what it's doing to console? */
    protected boolean m_isVerbose = false;
    
    public J48SS() {
    }
    
    public String globalInfo() {
        return "Class for generating a pruned or unpruned decision tree handling numeric, continuous, time series and" +
                " sequential attribute types. The algorithm is based on J48 code. For more information, see\n\n" +
                this.getTechnicalInformation().toString();
    }
    
    /**
     * Returns an instance of a TechnicalInformation object, containing
     * detailed information about the technical background of this class,
     * e.g., paper reference or book this class is based on.
     *
     * @return the technical information about this class
     */
    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation result = new TechnicalInformation(Type.BOOK);
        result.setValue(Field.AUTHOR, "Ross Quinlan");
        result.setValue(Field.YEAR, "1993");
        result.setValue(Field.TITLE, "C4.5: Programs for Machine Learning");
        result.setValue(Field.PUBLISHER, "Morgan Kaufmann Publishers");
        result.setValue(Field.ADDRESS, "San Mateo, CA");
        return result;
    }
    
    /**
     * Returns default capabilities of the classifier.
     *
     * @return the capabilities of this classifier
     */
    public Capabilities getCapabilities() {
        Capabilities result = new Capabilities(this);
        result.disableAll();
        result.enable(Capability.NOMINAL_ATTRIBUTES);
        result.enable(Capability.NUMERIC_ATTRIBUTES);
        result.enable(Capability.DATE_ATTRIBUTES);
        result.enable(Capability.MISSING_VALUES);
        result.enable(Capability.STRING_ATTRIBUTES);
        result.enable(Capability.NOMINAL_CLASS);
        result.enable(Capability.MISSING_CLASS_VALUES);
        result.setMinimumNumberInstances(0);
        return result;
    }
    
    /**
     * Generates the classifier.
     *
     * @param instances the data to train the classifier with
     *
     * @throws Exception if classifier can't be built successfully
     */
    public void buildClassifier(Instances instances) throws Exception {
        if (m_isVerbose) {
            System.out.println("Setting random seed to: " + m_Seed);
        }
        PseudoRandom.setRandomGenerator(m_Seed);
        Map<String, String> itemTranslationToVGEN   = new HashMap<>();
        Map<String, String> itemTranslationFromVGEN = new HashMap<>();
        
        // Process sequential attributes
        for (int i = 0; i < instances.numAttributes(); i++) {
            int firstFreeIndex = 0;
            if (i != instances.classIndex() && instances.attribute(i).isString()
                    && instances.attribute(i).name().startsWith("SEQ_")) {
                
                for (Instance inst : instances) {
                    String   instanceSequence = inst.stringValue(i).replace(" ", "");
                    String[] itemsets         = instanceSequence.split(">");
                    
                    for (String itemset : itemsets) {
                        String[] items = itemset.split(",");
                        
                        for (String item : items) {
                            if (! itemTranslationToVGEN.containsKey(i + "|" + item)) {
                                itemTranslationToVGEN.put(i + "|" + item, i + "|" + firstFreeIndex);
                                itemTranslationFromVGEN.put(i + "|" + firstFreeIndex, i + "|" + item);
                                firstFreeIndex++;
                            }
                        }
                    }
                }
            }
        }
        
        ModelSelection modSelection;
        if (m_binarySplits) {
            modSelection = new BinC45ModelSelection(m_minNumObj, instances, m_useMDLcorrection,
                    m_doNotMakeSplitPointActualValue, m_minimumSupport, m_useIGPruning, m_maxGap,
                    m_maxPatternLength + 1, m_patternWeight, m_popSize, m_numEvals, m_crossoverP, m_mutationP,
                    m_isVerbose);
            ((BinC45ModelSelection)modSelection).setItemTranslationToVGEN(itemTranslationToVGEN);
            ((BinC45ModelSelection)modSelection).setItemTranslationFromVGEN(itemTranslationFromVGEN);
        }
        else {
            modSelection = new C45ModelSelection(m_minNumObj, instances, m_useMDLcorrection,
                    m_doNotMakeSplitPointActualValue, m_minimumSupport, m_useIGPruning, m_maxGap,
                    m_maxPatternLength + 1, m_patternWeight, m_popSize, m_numEvals, m_crossoverP, m_mutationP,
                    m_isVerbose);
            ((C45ModelSelection)modSelection).setItemTranslationToVGEN(itemTranslationToVGEN);
            ((C45ModelSelection)modSelection).setItemTranslationFromVGEN(itemTranslationFromVGEN);
        }
        
        if (! m_reducedErrorPruning) {
            m_root = new C45PruneableClassifierTree(modSelection, ! m_unpruned, m_CF, m_subtreeRaising, ! m_noCleanup,
                    m_collapseTree);
        }
        else {
            m_root = new PruneableClassifierTree(modSelection, ! m_unpruned, m_numFolds, ! m_noCleanup, m_Seed);
        }
        
        m_root.buildClassifier(instances);
        if (m_binarySplits) {
            assert modSelection instanceof BinC45ModelSelection;
            ((BinC45ModelSelection)modSelection).cleanup();
        }
        else {
            assert modSelection instanceof C45ModelSelection;
            ((C45ModelSelection)modSelection).cleanup();
        }
    }
    
    /**
     * Alternative loader for TimeSeriesInstances objects
     *
     * @param instances Collection of TimeSeriesInstance objects
     *
     * @throws Exception From creating trees
     */
    public void buildClassifier(TimeSeriesInstances instances) throws Exception {
        //            ModelSelection modSelection;
        //            if (m_binarySplits) {
        //                modSelection = new BinC45ModelSelection(m_minNumObj, instances, m_useMDLcorrection,
        //                        m_doNotMakeSplitPointActualValue, m_minimumSupport, m_useIGPruning, m_maxGap,
        //                        m_maxPatternLength + 1, m_patternWeight, m_popSize, m_numEvals, m_Seed,
        //                        m_crossoverP, m_mutationP);
        //            }
        //            else {
        //                modSelection = new C45ModelSelection(m_minNumObj, instances, m_useMDLcorrection,
        //                        m_doNotMakeSplitPointActualValue, m_minimumSupport, m_useIGPruning,
        //                        m_maxGap, m_maxPatternLength + 1, m_patternWeight,
        //                        this.popSize, this.numEvals, m_Seed, this.crossoverP, this.mutationP);
        //            }
        //
        //            // Designate the type of tree depending on pruning value
        //            if (! m_reducedErrorPruning) {
        //                m_root = new C45PruneableClassifierTree(modSelection, ! m_unpruned, m_CF,
        //                        m_subtreeRaising, ! m_noCleanup,
        //                        m_collapseTree);
        //            }
        //            else {
        //                m_root = new PruneableClassifierTree(modSelection, ! m_unpruned, m_numFolds,
        //                        ! m_noCleanup, m_Seed);
        //            }
        //
        //
        //            m_root.buildClassifier(instances);
        //            if (m_binarySplits) {
        //                assert modSelection instanceof BinC45ModelSelection;
        //                ((BinC45ModelSelection)modSelection).cleanup();
        //            }
        //            else {
        //                assert modSelection instanceof C45ModelSelection;
        //                ((C45ModelSelection)modSelection).cleanup();
        //            }
    }
    
    /**
     * Classifies an instance.
     *
     * @param instance the instance to classify
     *
     * @return the classification for the instance
     * @throws Exception if instance can't be classified successfully
     */
    public double classifyInstance(Instance instance) throws Exception {
        return m_root.classifyInstance(instance);
    }
    
    /**
     * Returns class probabilities for an instance.
     *
     * @param instance the instance to calculate the class probabilities for
     *
     * @return the class probabilities
     * @throws Exception if distribution can't be computed successfully
     */
    public final double[] distributionForInstance(Instance instance) throws Exception {
        return m_root.distributionForInstance(instance, m_useLaplace);
    }
    
    /**
     * Returns the type of graph this classifier
     * represents.
     *
     * @return Drawable.TREE
     */
    public int graphType() { return 1; }
    
    /**
     * Returns graph describing the tree.
     *
     * @return the graph describing the tree
     * @throws Exception if graph can't be computed
     */
    public String graph() throws Exception { return m_root.graph(); }
    
    /**
     * Returns tree in prefix order.
     *
     * @return the tree in prefix order
     * @throws Exception if something goes wrong
     */
    public String prefix() throws Exception { return m_root.prefix(); }
    
    /**
     * Returns tree as an if-then statement.
     *
     * @param className the name of the Java class
     *
     * @return the tree as a Java if-then type statement
     * @throws Exception if something goes wrong
     */
    public String toSource(String className) throws Exception {
        StringBuilder[] source = m_root.toSource(className);
        return "class " + className + " {\n\n" + "  public static double classify(Object[] i)\n" + "    throws " +
                "Exception {\n\n" + "    double p = Double.NaN;\n" + source[0] + "    return p;\n" + "  }\n" +
                source[1] + "}\n";
    }
    
    public Enumeration<Option> listOptions() {
        Vector<Option> newVector = new Vector<>(13);
        newVector.addElement(new Option("\tUse unpruned tree.", "U", 0, "-U"));
        newVector.addElement(new Option("\tDo not collapse tree.", "O", 0, "-O"));
        newVector.addElement(new Option("\tSet confidence threshold for pruning.\n\t(default 0.25)", "C", 1,
                "-C <pruning confidence>"));
        newVector.addElement(new Option("\tSet minimum number of instances per leaf.\n\t(default 2)", "M", 1,
                "-M <minimum number of instances>"));
        newVector.addElement(new Option("\tUse reduced error pruning.", "R", 0, "-R"));
        newVector.addElement(new Option(
                "\tSet number of folds for reduced error\n\tpruning. One fold is used as pruning set.\n\t(default 3)",
                "N", 1, "-N <number of folds>"));
        newVector.addElement(new Option("\tUse binary splits only.", "B", 0, "-B"));
        newVector.addElement(new Option("\tDo not perform subtree raising.", "S", 0, "-S"));
        newVector.addElement(new Option("\tDo not clean up after the tree has been built.", "L", 0, "-L"));
        newVector.addElement(new Option("\tLaplace smoothing for predicted probabilities.", "A", 0, "-A"));
        newVector.addElement(new Option("\tDo not use MDL correction for info gain on numeric attributes.", "J", 0,
                "-J"));
        newVector.addElement(new Option("\tSeed for random data shuffling (default 1).", "Q", 1, "-Q <seed>"));
        newVector.addElement(new Option("\tDo not make split point actual value.", "-SP", 0, "-SP"));
        newVector.addElement(new Option("\tMinimum support for the pattern extraction algorithm, %.", "P", 1,
                "-P <minimum support %>"));
        newVector.addAll(Collections.list(super.listOptions())); // add all the possible options from AbstractClassifier
        return newVector.elements();
    }
    
    /**
     * Parses a given list of options.
     *
     * <!-- options-start -->
     * Valid options are:
     *
     * <pre> -U
     *  Use unpruned tree.</pre>
     *
     * <pre> -O
     *  Do not collapse tree.</pre>
     *
     * <pre> -C &lt;pruning confidence&gt;
     *  Set confidence threshold for pruning.
     *  (default 0.25)</pre>
     *
     * <pre> -M &lt;minimum number of instances&gt;
     *  Set minimum number of instances per leaf.
     *  (default 2)</pre>
     *
     * <pre> -R
     *  Use reduced error pruning.</pre>
     *
     * <pre> -N &lt;number of folds&gt;
     *  Set number of folds for reduced error
     *  pruning. One fold is used as pruning set.
     *  (default 3)</pre>
     *
     * <pre> -B
     *  Use binary splits only.</pre>
     *
     * <pre> -S
     *  Don't perform subtree raising.</pre>
     *
     * <pre> -L
     *  Do not clean up after the tree has been built.</pre>
     *
     * <pre> -A
     *  Laplace smoothing for predicted probabilities.</pre>
     *
     * <pre> -J
     *  Do not use MDL correction for info gain on numeric attributes.</pre>
     *
     * <pre> -Q &lt;seed&gt;
     *  Seed for random data shuffling (default 1).</pre>
     *
     * <pre> -Z
     *  Use IG scoring of the best non-sequential attribute to guide the pruning of pattern search space.</pre>
     *
     * <pre> -SP
     *  If given, do not make the split point the actual value.</pre>
     *
     * <pre> -P
     *  The minimum support that the extracted sequential patterns must have (between 0 and 1).</pre>
     *
     * <pre> -G
     *  The maximum gap between two itemsets in a pattern.</pre>
     *
     * <pre> -I
     *  The maximum pattern length in terms of itemset count.</pre>
     *
     * <pre> -W
     *  The weight used to evaluate the patterns.</pre>
     *
     * <pre> -POP
     *  The population size of the genetic algorithm i.e., the number of individuals.</pre>
     *
     * <pre> -EV
     *  The number of evaluations that are going to be carried out in the optimization process.</pre>
     *
     * <pre> -CR
     *  The probability of combining two individuals of the population.</pre>
     *
     * <pre> -MUT
     *  The probability of an element to undergo a random mutation.</pre>
     *
     * <!-- options-end -->
     *
     * @param options the list of options as an array of strings
     *
     * @throws Exception if an option is not supported
     */
    public void setOptions(String[] options) throws Exception {
        String minNumString = Utils.getOption('M', options);
        if (minNumString.length() != 0) {
            m_minNumObj = Integer.parseInt(minNumString);
        }
        else {
            m_minNumObj = 2;
        }
        
        m_binarySplits                   = Utils.getFlag('B', options);
        m_useLaplace                     = Utils.getFlag('A', options);
        m_useMDLcorrection               = ! Utils.getFlag('J', options);
        m_unpruned                       = Utils.getFlag('U', options);
        m_collapseTree                   = ! Utils.getFlag('O', options);
        m_subtreeRaising                 = ! Utils.getFlag('S', options);
        m_noCleanup                      = Utils.getFlag('L', options);
        m_useIGPruning                   = Utils.getFlag('Z', options);
        m_doNotMakeSplitPointActualValue = Utils.getFlag("SP", options);
        
        if (m_unpruned && ! m_subtreeRaising) {
            throw new Exception("Subtree raising doesn't need to be unset for unpruned tree!");
        }
        else {
            m_reducedErrorPruning = Utils.getFlag('R', options);
            if (m_unpruned && m_reducedErrorPruning) {
                throw new Exception("Unpruned tree and reduced error pruning can't be selected simultaneously!");
            }
            else {
                String confidenceString = Utils.getOption('C', options);
                if (confidenceString.length() != 0) {
                    if (m_reducedErrorPruning) {
                        throw new Exception("Setting the confidence doesn't make sense for reduced error pruning.");
                    }
                    
                    if (m_unpruned) {
                        throw new Exception("Doesn't make sense to change confidence for unpruned tree!");
                    }
                    
                    m_CF = new Float(confidenceString);
                    if (m_CF <= 0.0F || m_CF >= 1.0F) {
                        throw new Exception("Confidence has to be greater than zero and smaller than one!");
                    }
                }
                else {
                    m_CF = 0.25F;
                }
                
                String numFoldsString = Utils.getOption('N', options);
                if (numFoldsString.length() != 0) {
                    if (! m_reducedErrorPruning) {
                        throw new Exception("Setting the number of folds doesn't make sense if reduced error pruning" +
                                " is not selected.");
                    }
                    
                    m_numFolds = Integer.parseInt(numFoldsString);
                }
                else {
                    m_numFolds = 3;
                }
                
                String seedString = Utils.getOption('Q', options);
                if (seedString.length() != 0) {
                    m_Seed = Integer.parseInt(seedString);
                }
                else {
                    m_Seed = 1;
                }
                
                String supportString = Utils.getOption('P', options);
                if (supportString.length() != 0) {
                    m_minimumSupport = Double.parseDouble(supportString);
                }
                else {
                    m_minimumSupport = 0.5D;
                }
                
                String gapString = Utils.getOption('G', options);
                if (gapString.length() != 0) {
                    m_maxGap = Integer.parseInt(gapString);
                }
                else {
                    m_maxGap = 2;
                }
                
                String lengthString = Utils.getOption('I', options);
                if (lengthString.length() != 0) {
                    m_maxPatternLength = Integer.parseInt(lengthString);
                }
                else {
                    m_maxPatternLength = 20;
                }
                
                String weightString = Utils.getOption('W', options);
                if (weightString.length() != 0) {
                    m_patternWeight = Double.parseDouble(weightString);
                }
                else {
                    m_patternWeight = 0.75D;
                }
                
                String populationString = Utils.getOption("POP", options);
                if (populationString.length() != 0) {
                    m_popSize = Integer.parseInt(populationString);
                }
                else {
                    m_popSize = 100;
                }
                
                String evaluationsString = Utils.getOption("EV", options);
                if (evaluationsString.length() != 0) {
                    m_numEvals = Integer.parseInt(evaluationsString);
                }
                else {
                    m_numEvals = 500;
                }
                
                String crossoverString = Utils.getOption("CR", options);
                if (crossoverString.length() != 0) {
                    m_crossoverP = Double.parseDouble(crossoverString);
                }
                else {
                    m_crossoverP = 0.8D;
                }
                
                String mutationString = Utils.getOption("MUT", options);
                if (mutationString.length() != 0) {
                    m_mutationP = Double.parseDouble(mutationString);
                }
                else {
                    m_mutationP = 0.1D;
                }
                
                super.setOptions(options);
                Utils.checkForRemainingOptions(options);
            }
        }
    }
    
    /**
     * Gets the current settings of the Classifier.
     *
     * @return an array of strings suitable for passing to setOptions
     */
    public String[] getOptions() {
        Vector<String> options = new Vector<>();
        if (m_noCleanup) {
            options.add("-L");
        }
        
        if (! m_collapseTree) {
            options.add("-O");
        }
        
        if (m_unpruned) {
            options.add("-U");
        }
        else {
            if (! m_subtreeRaising) {
                options.add("-S");
            }
            
            if (m_reducedErrorPruning) {
                options.add("-R");
                options.add("-N");
                options.add("" + m_numFolds);
                options.add("-Q");
                options.add("" + m_Seed);
            }
            else {
                options.add("-C");
                options.add("" + m_CF);
            }
        }
        
        if (m_binarySplits) {
            options.add("-B");
        }
        
        if (m_useLaplace) {
            options.add("-A");
        }
        
        if (! m_useMDLcorrection) {
            options.add("-J");
        }
        
        if (m_useIGPruning) {
            options.add("-Z");
        }
        
        if (m_doNotMakeSplitPointActualValue) {
            options.add("-SP");
        }
        
        options.add("-M");
        options.add("" + m_minNumObj);
        options.add("-P");
        options.add("" + m_minimumSupport);
        options.add("-G");
        options.add("" + m_maxGap);
        options.add("-I");
        options.add("" + m_maxPatternLength);
        options.add("-W");
        options.add("" + m_patternWeight);
        
        options.add("-POP");
        options.add("" + m_popSize);
        options.add("-EV");
        options.add("" + m_numEvals);
        options.add("-CR");
        options.add("" + m_crossoverP);
        options.add("-MUT");
        options.add("" + m_mutationP);
        Collections.addAll(options, super.getOptions());
        return options.toArray(new String[0]);
    }
    
    /**
     * Returns the tip text for this property
     *
     * @return tip text for this property suitable for
     * displaying in the explorer/experimenter gui
     */
    public String seedTipText() {
        return "The seed used for randomizing the data when reduced-error pruning is used.";
    }
    
    /**
     * Get the value of Seed.
     *
     * @return Value of Seed.
     */
    public int getSeed() { return m_Seed; }
    
    /**
     * Set the value of Seed.
     *
     * @param newSeed Value to assign to Seed.
     */
    public void setSeed(int newSeed) { m_Seed = newSeed; }
    
    /**
     * Returns the tip text for this property
     *
     * @return tip text for this property suitable for
     * displaying in the explorer/experimenter gui
     */
    public String useLaplaceTipText() { return "Whether counts at leaves are smoothed based on Laplace."; }
    
    /**
     * Get the value of useLaplace.
     *
     * @return Value of useLaplace.
     */
    public boolean getUseLaplace() { return m_useLaplace; }
    
    /**
     * Set the value of useLaplace.
     *
     * @param newuseLaplace Value to assign to useLaplace.
     */
    public void setUseLaplace(boolean newuseLaplace) { m_useLaplace = newuseLaplace; }
    
    /**
     * Returns the tip text for this property
     *
     * @return tip text for this property suitable for
     * displaying in the explorer/experimenter gui
     */
    public String useMDLcorrectionTipText() {
        return "Whether MDL correction is used when finding splits on numeric attributes.";
    }
    
    /**
     * Get the value of useMDLcorrection.
     *
     * @return Value of useMDLcorrection.
     */
    public boolean getUseMDLcorrection() { return m_useMDLcorrection; }
    
    /**
     * Set the value of useMDLcorrection.
     *
     * @param newuseMDLcorrection Value to assign to useMDLcorrection.
     */
    public void setUseMDLcorrection(boolean newuseMDLcorrection) { m_useMDLcorrection = newuseMDLcorrection; }
    
    /**
     * Returns a description of the classifier.
     *
     * @return a description of the classifier
     */
    public String toString() {
        if (m_root == null) {
            return "No classifier built";
        }
        else {
            return m_unpruned ? "J48_IG unpruned tree\n------------------\n" + m_root :
                    "J48_IG pruned tree\n------------------\n" + m_root;
        }
    }
    
    /**
     * Returns a superconcise version of the model
     *
     * @return a summary of the model
     */
    public String toSummaryString() {
        return "Number of leaves: " + m_root.numLeaves() + "\n" + "Size of the tree: " + m_root
                .numNodes() + "\n";
    }
    
    /**
     * Returns the size of the tree
     *
     * @return the size of the tree
     */
    public double measureTreeSize() { return m_root.numNodes(); }
    
    /**
     * Returns the number of leaves
     *
     * @return the number of leaves
     */
    public double measureNumLeaves() { return m_root.numLeaves(); }
    
    /**
     * Returns the number of rules (same as number of leaves)
     *
     * @return the number of rules
     */
    public double measureNumRules() { return m_root.numLeaves(); }
    
    /**
     * Returns an enumeration of the additional measure names
     *
     * @return an enumeration of the measure names
     */
    public Enumeration<String> enumerateMeasures() {
        Vector<String> newVector = new Vector<>(3);
        newVector.addElement("measureTreeSize");
        newVector.addElement("measureNumLeaves");
        newVector.addElement("measureNumRules");
        return newVector.elements();
    }
    
    /**
     * Returns the value of the named measure
     *
     * @param additionalMeasureName the name of the measure to query for its value
     *
     * @return the value of the named measure
     * @throws IllegalArgumentException if the named measure is not supported
     */
    public double getMeasure(String additionalMeasureName) {
        if (additionalMeasureName.compareToIgnoreCase("measureNumRules") == 0) {
            return this.measureNumRules();
        }
        else if (additionalMeasureName.compareToIgnoreCase("measureTreeSize") == 0) {
            return this.measureTreeSize();
        }
        else if (additionalMeasureName.compareToIgnoreCase("measureNumLeaves") == 0) {
            return this.measureNumLeaves();
        }
        else {
            throw new IllegalArgumentException(additionalMeasureName + " not supported (j48_IG)");
        }
    }
    
    /**
     * Returns the tip text for this property
     *
     * @return tip text for this property suitable for
     * displaying in the explorer/experimenter gui
     */
    public String unprunedTipText() {
        return "Whether pruning is performed.";
    }
    
    /**
     * Get the value of unpruned.
     *
     * @return Value of unpruned.
     */
    public boolean getUnpruned() { return m_unpruned; }
    
    /**
     * Set the value of unpruned. Turns reduced-error pruning
     * off if set.
     *
     * @param v Value to assign to unpruned.
     */
    public void setUnpruned(boolean v) {
        m_reducedErrorPruning = ! v;
        m_unpruned            = v;
    }
    
    /**
     * Returns the tip text for this property
     *
     * @return tip text for this property suitable for
     * displaying in the explorer/experimenter gui
     */
    public String collapseTreeTipText() { return "Whether parts are removed that do not reduce training error."; }
    
    /**
     * Get the value of collapseTree.
     *
     * @return Value of collapseTree.
     */
    public boolean getCollapseTree() { return m_collapseTree; }
    
    /**
     * Set the value of collapseTree.
     *
     * @param v Value to assign to collapseTree.
     */
    public void setCollapseTree(boolean v) { m_collapseTree = v; }
    
    /**
     * Returns the tip text for this property
     *
     * @return tip text for this property suitable for
     * displaying in the explorer/experimenter gui
     */
    public String confidenceFactorTipText() {
        return "The confidence factor used for pruning (smaller values incur more pruning).";
    }
    
    /**
     * Get the value of CF.
     *
     * @return Value of CF.
     */
    public float getConfidenceFactor() { return m_CF; }
    
    /**
     * Set the value of CF.
     *
     * @param v Value to assign to CF.
     */
    public void setConfidenceFactor(float v) { m_CF = v; }
    
    /**
     * Returns the tip text for this property
     *
     * @return tip text for this property suitable for
     * displaying in the explorer/experimenter gui
     */
    public String minimumSupportTipText() {
        return "The fractional minimum support used for the pattern extraction algorithm.";
    }
    
    /**
     * Get the value of minimumSupport.
     *
     * @return Value of minimumSupport.
     */
    public double getMinimumSupport() { return m_minimumSupport; }
    
    /**
     * Set the value of minimumSupport.
     *
     * @param v Value to assign to minimumSupport.
     */
    public void setMinimumSupport(double v) { m_minimumSupport = v; }
    
    /**
     * Get the value of maxGap.
     *
     * @return Value of maxGap.
     */
    public int getMaxGap() { return m_maxGap; }
    
    /**
     * Set the value of maxGap.
     *
     * @param v Value to assign to maxGap.
     */
    public void setMaxGap(int v) { m_maxGap = v; }
    
    /**
     * Get the value of maxPatternLength.
     *
     * @return Value of maxPatternLength.
     */
    public int getMaxPatternLength() { return m_maxPatternLength; }
    
    /**
     * Set the value of maxPatternLength.
     *
     * @param v Value to assign to maxPatternLength.
     */
    public void setMaxPatternLength(int v) { m_maxPatternLength = v; }
    
    /**
     * Get the value of patternWeight.
     *
     * @return Value of patternWeight.
     */
    public double getPatternWeight() { return m_patternWeight; }
    
    /**
     * Set the value of patternWeight.
     *
     * @param v Value to assign to patternWeight.
     */
    public void setPatternWeight(double v) { m_patternWeight = v; }
    
    /**
     * Get the value of useIGPruning.
     *
     * @return Value of useIGPruning.
     */
    public boolean getUseIGPruning() { return m_useIGPruning; }
    
    /**
     * Set the value of useIGPruning.
     *
     * @param v Value to assign to useIGPruning.
     */
    public void setUseIGPruning(boolean v) { m_useIGPruning = v; }
    
    /**
     * Returns the tip text for this property
     *
     * @return tip text for this property suitable for
     * displaying in the explorer/experimenter gui
     */
    public String minNumObjTipText() {
        return "The minimum number of instances per leaf.";
    }
    
    /**
     * Get the value of minNumObj.
     *
     * @return Value of minNumObj.
     */
    public int getMinNumObj() {
        return m_minNumObj;
    }
    
    /**
     * Set the value of minNumObj.
     *
     * @param v Value to assign to minNumObj.
     */
    public void setMinNumObj(int v) {
        m_minNumObj = v;
    }
    
    /**
     * Returns the tip text for this property
     *
     * @return tip text for this property suitable for
     * displaying in the explorer/experimenter gui
     */
    public String reducedErrorPruningTipText() {
        return "Whether reduced-error pruning is used instead of Error-Based pruning.";
    }
    
    /**
     * Get the value of reducedErrorPruning.
     *
     * @return Value of reducedErrorPruning.
     */
    public boolean getReducedErrorPruning() {
        return m_reducedErrorPruning;
    }
    
    /**
     * Set the value of reducedErrorPruning. Turns
     * unpruned trees off if set.
     *
     * @param v Value to assign to reducedErrorPruning.
     */
    public void setReducedErrorPruning(boolean v) {
        if (v) {
            m_unpruned = false;
        }
        
        m_reducedErrorPruning = v;
    }
    
    /**
     * Returns the tip text for this property
     *
     * @return tip text for this property suitable for
     * displaying in the explorer/experimenter gui
     */
    public String numFoldsTipText() {
        return "Determines the amount of data used for reduced-error pruning.  One fold is used for pruning, the rest" +
                " for growing the tree.";
    }
    
    /**
     * Get the value of numFolds.
     *
     * @return Value of numFolds.
     */
    public int getNumFolds() {
        return m_numFolds;
    }
    
    /**
     * Set the value of numFolds.
     *
     * @param v Value to assign to numFolds.
     */
    public void setNumFolds(int v) {
        m_numFolds = v;
    }
    
    /**
     * Returns the tip text for this property
     *
     * @return tip text for this property suitable for
     * displaying in the explorer/experimenter gui
     */
    public String binarySplitsTipText() {
        return "Whether to use binary splits on nominal attributes when building the trees.";
    }
    
    /**
     * Get the value of binarySplits.
     *
     * @return Value of binarySplits.
     */
    public boolean getBinarySplits() {
        return m_binarySplits;
    }
    
    /**
     * Set the value of binarySplits.
     *
     * @param v Value to assign to binarySplits.
     */
    public void setBinarySplits(boolean v) {
        m_binarySplits = v;
    }
    
    /**
     * Returns the tip text for this property
     *
     * @return tip text for this property suitable for
     * displaying in the explorer/experimenter gui
     */
    public String subtreeRaisingTipText() {
        return "Whether to consider the subtree raising operation when pruning.";
    }
    
    /**
     * Get the value of subtreeRaising.
     *
     * @return Value of subtreeRaising.
     */
    public boolean getSubtreeRaising() {
        return m_subtreeRaising;
    }
    
    /**
     * Set the value of subtreeRaising.
     *
     * @param v Value to assign to subtreeRaising.
     */
    public void setSubtreeRaising(boolean v) {
        m_subtreeRaising = v;
    }
    
    /**
     * Returns the tip text for this property
     *
     * @return tip text for this property suitable for
     * displaying in the explorer/experimenter gui
     */
    public String saveInstanceDataTipText() { return "Whether to save the training data for visualization."; }
    
    /**
     * Check whether instance data is to be saved.
     *
     * @return true if instance data is saved
     */
    public boolean getSaveInstanceData() { return m_noCleanup; }
    
    /**
     * Set whether instance data is to be saved.
     *
     * @param v true if instance data is to be saved
     */
    public void setSaveInstanceData(boolean v) { m_noCleanup = v; }
    
    /**
     * Returns the tip text for this property
     *
     * @return tip text for this property suitable for
     * displaying in the explorer/experimenter gui
     */
    public String doNotMakeSplitPointActualValueTipText() {
        return "If true, the split point is not relocated to an actual data value. This can yield substantial " +
                "speed-ups for large datasets with numeric attributes.";
    }
    
    /**
     * Get the value of doNotMakeSplitPointActualValue.
     *
     * @return Value of doNotMakeSplitPointActualValue.
     */
    public boolean getDoNotMakeSplitPointActualValue() { return m_doNotMakeSplitPointActualValue; }
    
    /**
     * Set the value of doNotMakeSplitPointActualValue.
     *
     * @param v Value to assign to doNotMakeSplitPointActualValue.
     */
    public void setDoNotMakeSplitPointActualValue(boolean v) { m_doNotMakeSplitPointActualValue = v; }
    
    /**
     * Get the value of popSize.
     *
     * @return Value of popSize.
     */
    public int getPopSize() { return m_popSize; }
    
    /**
     * Set the value of popSize.
     *
     * @param v Value to assign to popSize.
     */
    public void setPopSize(int v) { m_popSize = v; }
    
    /**
     * Get the value of numEvals.
     *
     * @return Value of numEvals.
     */
    public int getM_numEvals() { return m_numEvals; }
    
    /**
     * Set the value of numEvals.
     *
     * @param v Value to assign to numEvals.
     */
    public void setM_numEvals(int v) { m_numEvals = v; }
    
    /**
     * Get the value of crossoverP.
     *
     * @return Value of crossoverP.
     */
    public double getM_crossoverP() { return m_crossoverP; }
    
    /**
     * Set the value of crossoverP.
     *
     * @param v Value to assign to crossoverP.
     */
    public void setCrossoverP(double v) { m_crossoverP = v; }
    
    /**
     * Get the value of mutationP.
     *
     * @return Value of mutationP.
     */
    public double getMutationP() { return m_mutationP; }
    
    /**
     * Set the value of mutationP.
     *
     * @param v Value to assign to mutationP.
     */
    public void setMutationP(double v) { m_mutationP = v; }
    
    /**
     * Returns the revision string.
     */
    public String getRevision() { return RevisionUtils.extract("$Revision: 11194 $"); }
    
    /**
     * Builds the classifier to generate a partition.
     */
    public void generatePartition(Instances data) throws Exception { this.buildClassifier(data); }
    
    /**
     * Computes an array that indicates node membership.
     */
    public double[] getMembershipValues(Instance inst) { return m_root.getMembershipValues(inst); }
    
    /**
     * Returns the number of elements in the partition.
     */
    public int numElements() { return m_root.numNodes(); }
    
    /**
     * Set whether the classifier prints out information such as the seed EA, IG and instances read per round of NSGA-II
     *
     * @param v value to assign to verbosity
     */
    public void setVerbosity(boolean v) { m_isVerbose = v; }
    
    /**
     * Main method for testing this class
     *
     * @param argv the commandline options
     */
    public static void main(String[] argv){
        runClassifier(new J48SS(), argv);
    }
}
