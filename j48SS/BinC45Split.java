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

package weka.classifiers.trees.j48SS;

import weka.classifiers.trees.j48SS.jmetal.PNSGAIITimeSeries;
import weka.classifiers.trees.j48SS.spmf.AlgoVGEN;
import weka.core.*;

import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;


/**
 * Class implementing a binary C4.5-like split on an attribute.
 *
 * @author Eibe Frank (eibe@cs.waikato.ac.nz)
 * @version $Revision: 8034 $
 */
public class BinC45Split extends ClassifierSplitModel {
    /** for serialization */
    private static final long serialVersionUID = - 1278776919563022474L;
    
    /** Attribute to split on. */
    private final int m_attIndex;
    
    /** Minimum number of instances that have to occur in at least two subsets induced by split. */
    protected final int m_minNoObj;
    
    /** Use MDL correction when finding splits on numeric attributes? */
    protected final boolean m_useMDLcorrection;
    
    /** Value of split point. */
    private double m_splitPoint;
    
    /** InfoGain of split. */
    private double m_infoGain;
    
    /** The sum of the weights of the instances. */
    protected final double m_sumOfWeights;
    
    /** Static reference to splitting criterion. */
    protected static InfoGainSplitCrit m_infoGainCrit = new InfoGainSplitCrit();
    
    
    protected double[] m_prevFoundIG;
    
    /** The minimum support that the extracted sequential patterns must have (between 0 and 1) */
    protected double m_minimumSupport = 0.05D;
    
    /** The maximum gap between two itemsets in a pattern (1 = No gap) */
    protected int m_maxGap;
    
    /** The maximum pattern length in terms of itemset count */
    protected int m_maxPatternLength;
    
    /** The weight used to evaluate the patterns */
    protected double m_patternWeight;
    
    /** The population size of the genetic algorithm i.e., the number of individuals */
    private int m_popSize;
    
    /**
     * The number of evaluations that are going to be carried out in the optimization process
     * (should be higher than popSize)
     */
    private int m_numEvals;
    
    /** The probability of combining two individuals of the population */
    private double m_crossoverProb;
    
    /** The  probability of an element to undergo a random mutation */
    private double m_mutationProb;
    
    protected Map<String, String> m_toVGEN;
    protected Map<String, String> m_fromVGEN;
    
    private String   m_shapeletPatternValue;
    private double[] m_shapeletArray;
    
    /** Should the classifier print out what it's doing to console? */
    protected boolean m_isVerbose;
    
    /**
     * Initializes the split model for Sequential data.
     */
    public BinC45Split(
            int attIndex, int minNoObj, double sumOfWeights, boolean useMDLcorrection, double[] baseIG,
            double minimumSupport, Map<String, String> toVGEN, Map<String, String> fromVGEN, int maxGap,
            int maxPatLength, double patternWeight, boolean isVerbose) {
        m_attIndex         = attIndex;
        m_minNoObj         = minNoObj;
        m_sumOfWeights     = sumOfWeights;
        m_useMDLcorrection = useMDLcorrection;
        m_prevFoundIG      = baseIG.clone();
        m_minimumSupport   = minimumSupport;
        m_toVGEN           = toVGEN;
        m_fromVGEN         = fromVGEN;
        m_maxGap           = maxGap;
        m_maxPatternLength = maxPatLength;
        m_patternWeight    = patternWeight;
        m_isVerbose        = isVerbose;
    }
    
    /**
     * Initializes the split model for time series data.
     */
    public BinC45Split(
            int attIndex, int minNoObj, double sumOfWeights, boolean useMDLcorrection, int popSize, int numEvals,
            double crossoverP, double mutationP, double patternWeight, boolean isVerbose) {
        m_attIndex         = attIndex;
        m_minNoObj         = minNoObj;
        m_sumOfWeights     = sumOfWeights;
        m_useMDLcorrection = useMDLcorrection;
        m_prevFoundIG      = null;
        m_patternWeight    = patternWeight;
        m_popSize          = popSize;
        m_numEvals         = numEvals;
        m_crossoverProb    = crossoverP;
        m_mutationProb     = mutationP;
        m_isVerbose        = isVerbose;
    }
    
    /**
     * Initializes the basic split model.
     */
    public BinC45Split(int attIndex, int minNoObj, double sumOfWeights, boolean useMDLcorrection) {
        m_attIndex         = attIndex;
        m_minNoObj         = minNoObj;
        m_sumOfWeights     = sumOfWeights;
        m_useMDLcorrection = useMDLcorrection;
        m_prevFoundIG      = null;
    }
    
    /**
     * Creates a C4.5-type split on the given data.
     *
     * @throws Exception if something goes wrong
     */
    public void buildClassifier(Instances trainInstances) throws Exception {
        // Initialize the remaining instance variables.
        m_numSubsets = 0;
        m_splitPoint = Double.MAX_VALUE;
        m_infoGain   = 0.0D;
        
        // Different treatment for enumerated, numeric, sequential and time series attributes.
        if (trainInstances.attribute(m_attIndex).isNominal()) {
            handleEnumeratedAttribute(trainInstances);
        }
        else if (trainInstances.attribute(m_attIndex).isString() && trainInstances.attribute(m_attIndex)
                .name().startsWith("SEQ_")) {
            this.handleSequentialStringAttribute(trainInstances);
        }
        else if (trainInstances.attribute(m_attIndex).isString() && trainInstances.attribute(m_attIndex)
                .name().startsWith("TS_")) {
            this.handleTimeSeriesStringAttribute(trainInstances);
        }
        else if (trainInstances.attribute(m_attIndex).isNumeric()) {
            trainInstances.sort(trainInstances.attribute(m_attIndex));
            this.handleNumericAttribute(trainInstances);
        }
        else {
            System.err.println("Unexpected attribute");
            System.exit(1);
        }
    }
    
    /**
     * @return index of attribute for which split was generated.
     */
    public final int attIndex() { return m_attIndex; }
    
    /**
     * Returns the split point (numeric attribute only).
     *
     * @return the split point used for a test on a numeric attribute
     */
    public double splitPoint() { return m_splitPoint; }
    
    /**
     * Gets class probability for instance.
     */
    public final double classProb(int classIndex, Instance instance, int theSubset) {
        if (theSubset > - 1) {
            return Utils.gr(m_distribution.perBag(theSubset), 0.0D) ?
                    m_distribution.prob(classIndex, theSubset) : m_distribution.prob(classIndex);
        }
        else {
            double[] weights = this.weights(instance);
            if (weights == null) {
                return m_distribution.prob(classIndex);
            }
            else {
                double prob = 0.0D;
                for (int i = 0; i < weights.length; i++) {
                    prob += weights[i] * m_distribution.prob(classIndex, i);
                }
                return prob;
            }
        }
    }
    
    /**
     * Creates split on enumerated attribute.
     *
     * @param data data to create a distribution from
     */
    private void handleEnumeratedAttribute(Instances data) {
        int          numAttValues    = data.attribute(m_attIndex).numValues();
        Distribution newDistribution = new Distribution(numAttValues, data.numClasses());
        
        // Only Instances with known values are relevant.
        for (Instance inst : data) {
            if (! inst.isMissing(m_attIndex)) {
                newDistribution.add((int)inst.value(m_attIndex), inst);
            }
        }
        m_distribution = newDistribution;
        
        // For all values
        for (int i = 0; i < numAttValues; i++) {
            if (Utils.grOrEq(newDistribution.perBag(i), m_minNoObj)) {
                Distribution secondDistribution = new Distribution(newDistribution, i);
                
                // Check if minimum number of Instances in the two subsets.
                if (secondDistribution.check(m_minNoObj)) {
                    m_numSubsets = 2;
                    double currIG = m_infoGainCrit.splitCritValue(secondDistribution, m_sumOfWeights);
                    if (i == 0 || Utils.gr(currIG, m_infoGain)) {
                        m_infoGain     = currIG;
                        m_splitPoint   = i;
                        m_distribution = secondDistribution;
                    }
                }
            }
        }
    }
    
    /**
     * Creates split on time series Instances attribute.
     *
     * @param data data to be processed
     *
     * @throws Exception for pNSGAII when retrieving genetic solution
     */
    private void handleTimeSeriesStringAttribute(Instances data) throws Exception {
        // Current attribute is a string attribute encoding a time series. The new distribution is going to have
        // only 2 bags, according to the distance between the shapelet and the time series.
        
        // Call NSGA-II here. The output is a double array representing the selected shapelet, input is a list of
        // strings, where each element is made by: class,value_1,value_2,...
        List<String> instancesForEAList = new ArrayList<>();
        for (Instance instance : data) {
            if (! instance.isMissing(m_attIndex)) {
                String instanceString = instance.stringValue(instance.classIndex()) + "," +
                        instance.stringValue(m_attIndex).replace(" ", "");
                instancesForEAList.add(instanceString);
            }
        }
        
        double minSplitEA = 0.1D * (double)data.numInstances() / (double)data.numClasses();
        if (Utils.smOrEq(minSplitEA, m_minNoObj)) {
            minSplitEA = m_minNoObj;
        }
        else if (Utils.gr(minSplitEA, 25.0D)) {
            minSplitEA = 25.0D;
        }
        
        Map<String, double[]> results = PNSGAIITimeSeries.getGeneticSolution(minSplitEA, instancesForEAList,
                m_popSize, m_numEvals, m_crossoverProb, m_mutationProb, m_patternWeight, m_isVerbose);
        
        double   infoGainfromEA = ((double[])results.get("IG"))[0];
        double[] shapeletValue  = results.get("shapelet");
        
        // Was there any useful shapelet?
        if (Arrays.toString(shapeletValue).equals("null")) {
            return;
        }
        
        m_shapeletPatternValue = Arrays.toString(shapeletValue);
        m_shapeletArray        = shapeletValue;
        
        // Sort the instances according to the value they take on the distance from the shapelet
        Map<String, Double> seriesDistanceShapelet = new HashMap<>();
        
        for (Instance instance : data) {
            if (! (instance.isMissing(m_attIndex))) {
                String   tsString      = instance.stringValue(m_attIndex);
                String[] tsSplitString = tsString.split(",");
                double[] tsDouble      = new double[tsSplitString.length];
                
                for (int i = 0; i < tsSplitString.length; i++) {
                    tsDouble[i] = Double.parseDouble(tsSplitString[i]);
                }
                
                double currentInfoGain = this.subsequenceDistOpt(tsDouble, m_shapeletArray);
                seriesDistanceShapelet.put(tsString, currentInfoGain);
            }
        }
        
        sortTS(data, data.attribute(m_attIndex), seriesDistanceShapelet);
        
        //Now proceed as you would do with a numeric attribute
        m_distribution = new Distribution(2, data.numClasses());
        
        // Only Instances with known values are relevant.
        int instNum = 0;
        for (Instance instance : data) {
            if (instance.isMissing(m_attIndex)) {
                break;
            }
            m_distribution.add(1, instance);
            instNum++;
        }
        
        // This assignment works since before calling this method the instances are sorted
        // according to the value they take on the numeric attribute
        int firstMiss = instNum;
        
        // Compute minimum number of Instances required in each subset.
        double minSplit = 0.1D * m_distribution.total() / (double)data.numClasses();
        if (Utils.smOrEq(minSplit, m_minNoObj)) {
            minSplit = m_minNoObj;
        }
        else if (Utils.gr(minSplit, 25.0D)) {
            minSplit = 25.0D;
        }
        
        // Enough Instances with known values?
        if (Utils.sm(firstMiss, 2.0D * minSplit)) {
            return;
        }
        
        // Compute values of criteria for all possible split indices.
        int next       = 1;
        int last       = 0;
        int index      = 0;
        int splitIndex = - 1;
        for (double defaultEnt = m_infoGainCrit.oldEnt(m_distribution); next < firstMiss; next++) {
            if (seriesDistanceShapelet.get(data.instance(next - 1).stringValue(m_attIndex)) + 1.0E-5D <
                    seriesDistanceShapelet.get(data.instance(next).stringValue(m_attIndex))) {
                // Move class values for all Instances up to next possible split point.
                m_distribution.shiftRange(1, 0, data, last, next);
                
                // Check if enough Instances in each subset and compute values for criteria.
                if (Utils.grOrEq(m_distribution.perBag(0), minSplit) && Utils.grOrEq(m_distribution.perBag(1),
                        minSplit)) {
                    double currentInfoGain = m_infoGainCrit.splitCritValue(m_distribution, m_sumOfWeights, defaultEnt);
                    if (Utils.gr(currentInfoGain, m_infoGain)) {
                        m_infoGain = currentInfoGain;
                        splitIndex = next - 1;
                    }
                    index++;
                }
                last = next;
            }
        }
        
        // Was there any useful split?
        if (index == 0) {
            return;
        }
        
        // Compute modified information gain for best split.
        double saveIG = m_infoGain;
        if (m_useMDLcorrection) {
            m_infoGain -= Utils.log2(index) / m_sumOfWeights;
        }
        
        if (Utils.smOrEq(m_infoGain, 0.0D)) {
            return;
        }
        
        // Set instance variables' values to values for best split.
        m_numSubsets = 2;
        m_splitPoint = (seriesDistanceShapelet.get(data.instance(splitIndex + 1).stringValue(m_attIndex)) +
                seriesDistanceShapelet.get(data.instance(splitIndex).stringValue(m_attIndex))) / 2.0D;
        
        // In case we have a numerical precision problem we need to choose the smaller value
        if (m_splitPoint == seriesDistanceShapelet.get(data.instance(splitIndex + 1).stringValue(m_attIndex))) {
            this.m_splitPoint = seriesDistanceShapelet.get(data.instance(splitIndex).stringValue(m_attIndex));
        }
        
        // Restore distribution for best split.
        m_distribution = new Distribution(2, data.numClasses());
        m_distribution.addRange(0, data, 0, splitIndex + 1);
        m_distribution.addRange(1, data, splitIndex + 1, firstMiss);
        
        if (m_isVerbose) {
            System.out.println("IG from EA vs IG from split: " + infoGainfromEA + " / " + saveIG);
        }
    }
    
    /**
     * Sorts time series attributes within an Instance object
     *
     * @param data                 object to sort
     * @param att                  Attribute to sort on
     * @param distanceFromShapelet map of distances from shapelet
     */
    private static void sortTS(Instances data, Attribute att, Map<String, Double> distanceFromShapelet) {
        int        attIndex = att.index();
        double[]   vals     = new double[data.numInstances()];
        Instance[] backup   = new Instance[vals.length];
        
        for (int i = 0; i < data.numInstances(); i++) {
            Instance inst = data.instance(i);
            backup[i] = inst;
            double val = inst.value(attIndex);
            if (Utils.isMissingValue(val)) {
                vals[i] = Double.MAX_VALUE;
            }
            else {
                String instanceString = inst.stringValue(attIndex);
                vals[i] = distanceFromShapelet.get(instanceString);
            }
        }
        
        int[] sortOrder = Utils.sortWithNoMissingValues(vals);
        
        for (int i = 0; i < vals.length; i++) {
            data.set(i, backup[sortOrder[i]]); // could be issues with this
        }
    }
    
    /**
     * Creates split on sequential Instances attribute.
     *
     * @param data data to look for sequential patterns
     */
    private void handleSequentialStringAttribute(Instances data) {
        m_distribution = new Distribution(2, data.numClasses());
        List<int[]>  horizontalDB = new ArrayList<>();
        List<String> instClasses  = new ArrayList<>();
        
        for (Instance instance : data) {
            instClasses.add(instance.stringValue(instance.classIndex()));
            String[]    splitPattern    = instance.stringValue(m_attIndex).replace(" ", "").split(">");
            List<int[]> itemsetAsArrays = new ArrayList<>();
            
            int totalItems = 0;
            for (String itemset : splitPattern) {
                String[] items = itemset.split(",");
                totalItems += items.length;
                int[] tempItemset = new int[items.length];
                
                for (int i = 0; i < items.length; i++) {
                    String anItem = items[i];
                    tempItemset[i] = Integer.parseInt((m_toVGEN.get(m_attIndex + "|" + anItem)).split("\\|")[1]);
                }
                Arrays.sort(tempItemset);
                itemsetAsArrays.add(tempItemset);
            }
            
            int[] transactionArray = new int[totalItems + 2 + (splitPattern.length - 1)];
            int   index            = 0;
            for (int[] itemset : itemsetAsArrays) {
                for (int item : itemset) {
                    transactionArray[index] = item;
                    index++;
                }
                transactionArray[index] = - 1;
            }
            transactionArray[index] = - 2;
            horizontalDB.add(transactionArray);
        }
        
        AlgoVGEN algoVGEN = new AlgoVGEN(horizontalDB, instClasses, m_patternWeight,
                m_isVerbose);
        if (m_maxGap > 0) {
            algoVGEN.setMaxGap(m_maxGap);
        }
        
        if (m_maxPatternLength > 0) {
            algoVGEN.setMaximumPatternLength(m_maxPatternLength);
        }
        
        Map<String, String> results = algoVGEN.runAlgorithm(m_prevFoundIG, m_minimumSupport);
        m_infoGain = Double.parseDouble(results.get("IG"));
        String[] splitPattern = results.get("pattern").split("-1");
        m_shapeletPatternValue = "";
        if (m_infoGain != - 1.0D) {
            for (String pattern : splitPattern) {
                String[] splitPatternElement = pattern.split(" ");
                
                for (String value : splitPatternElement) {
                    if (! value.equals("-2") && ! value.equals(" ") && value.length() > 0) {
                        m_shapeletPatternValue += (m_fromVGEN.get(m_attIndex + "|" + value)).split("\\|")[1] + ",";
                    }
                }
                m_shapeletPatternValue = m_shapeletPatternValue.substring(0, m_shapeletPatternValue.length() - 1) + ">";
            }
            
            m_shapeletPatternValue = m_shapeletPatternValue.substring(0, m_shapeletPatternValue.length() - 1);
            String[]              instancesIndexes = results.get("inst_indexes").split(" ");
            Map<Integer, Boolean> instancesMap     = new HashMap<>();
            
            for (String instIndex : instancesIndexes) {
                instancesMap.put(Integer.parseInt(instIndex), true);
            }
            
            int numKnownValues = 0;
            for (int i = 0; i < data.numInstances(); i++) {
                boolean  patternMatched = instancesMap.containsKey(i);
                Instance inst           = data.get(i);
                if (! inst.isMissing(m_attIndex)) {
                    if (patternMatched) {
                        m_distribution.add(1, inst);
                    }
                    else {
                        m_distribution.add(0, inst);
                    }
                    numKnownValues++;
                }
            }
            
            double minSplit = 0.1D * m_distribution.total() / (double)data.numClasses();
            if (Utils.smOrEq(minSplit, m_minNoObj)) {
                minSplit = m_minNoObj;
            }
            else if (Utils.gr(minSplit, 25.0D)) {
                minSplit = 25.0D;
            }
            
            if (! Utils.sm(numKnownValues, 2.0D * minSplit)) {
                if (! Utils.smOrEq(m_infoGain, 0.0D) && m_distribution.check(m_minNoObj)) {
                    m_numSubsets = 2;
                    m_splitPoint = - 1.0D;
                }
            }
        }
    }
    
    /**
     * Creates split on numeric attribute.
     *
     * @param data Data to be used to find the split
     */
    private void handleNumericAttribute(Instances data) {
        // Current attribute is a numeric attribute.
        m_distribution = new Distribution(2, data.numClasses());
        
        // Only Instances with known values are relevant.
        int instNum = 0;
        for (Instance instance : data) {
            if (instance.isMissing(m_attIndex)) {
                break;
            }
            m_distribution.add(1, instance);
            instNum++;
        }
        
        // Compute minimum number of Instances required in each subset.
        int    firstMiss = instNum;
        double minSplit  = 0.1D * m_distribution.total() / (double)data.numClasses();
        if (Utils.smOrEq(minSplit, m_minNoObj)) {
            minSplit = m_minNoObj;
        }
        else if (Utils.gr(minSplit, 25.0D)) {
            minSplit = 25.0D;
        }
        
        // Enough Instances with known values?
        if (Utils.sm(firstMiss, 2.0D * minSplit)) {
            return;
        }
        
        // Compute values of criteria for all possible split indices.
        int next       = 1;
        int last       = 0;
        int index      = 0;
        int splitIndex = - 1;
        for (double defaultEnt = m_infoGainCrit.oldEnt(m_distribution); next < firstMiss; next++) {
            if (data.instance(next - 1).value(m_attIndex) + 1.0E-5D < data.instance(next).value(m_attIndex)) {
                
                // Move class values for all Instances up to next possible split point.
                m_distribution.shiftRange(1, 0, data, last, next);
                
                // Check if enough Instances in each subset and compute values for criteria.
                if (Utils.grOrEq(m_distribution.perBag(0), minSplit) &&
                        Utils.grOrEq(m_distribution.perBag(1), minSplit)) {
                    double currentInfoGain = m_infoGainCrit.splitCritValue(m_distribution, m_sumOfWeights, defaultEnt);
                    if (Utils.gr(currentInfoGain, m_infoGain)) {
                        m_infoGain = currentInfoGain;
                        splitIndex = next - 1;
                    }
                    index++;
                }
                last = next;
            }
        }
        
        // Was there any useful split?
        if (index == 0) {
            return;
        }
        
        // Compute modified information gain for best split.
        if (m_useMDLcorrection) {
            m_infoGain -= Utils.log2(index) / m_sumOfWeights;
        }
        
        if (Utils.smOrEq(m_infoGain, 0.0D)) {
            return;
        }
        
        // Set instance variables' values to values for best split.
        m_numSubsets = 2;
        m_splitPoint = (data.instance(splitIndex + 1).value(m_attIndex) +
                data.instance(splitIndex).value(m_attIndex)) / 2.0D;
        
        // In case we have a numerical precision problem we need to choose the smaller value
        if (m_splitPoint == data.instance(splitIndex + 1).value(m_attIndex)) {
            m_splitPoint = data.instance(splitIndex).value(m_attIndex);
        }
        
        // Restore distribution for best split.
        m_distribution = new Distribution(2, data.numClasses());
        m_distribution.addRange(0, data, 0, splitIndex + 1);
        m_distribution.addRange(1, data, splitIndex + 1, firstMiss);
    }
    
    /**
     * @return (C4.5 - type) information gain for the generated split.
     */
    public final double infoGain() { return m_infoGain; }
    
    /**
     * Prints left side of condition.
     *
     * @param data the data to get the attribute name from.
     *
     * @return the attribute name
     */
    public final String leftSide(Instances data) {
        return data.attribute(m_attIndex).isString() && data.attribute(m_attIndex).name().startsWith("TS_") ?
                "d(" + data.attribute(m_attIndex).name() + "," : data.attribute(m_attIndex).name();
    }
    
    /**
     * Prints the condition satisfied by instances in a subset.
     *
     * @param index of subset and training set.
     */
    public final String rightSide(int index, Instances data) {
        StringBuilder text = new StringBuilder();
        if (data.attribute(m_attIndex).isNominal()) {
            if (index == 0) {
                text.append(" = ").append(data.attribute(m_attIndex).value((int)m_splitPoint));
            }
            else {
                text.append(" != ").append(data.attribute(m_attIndex).value((int)m_splitPoint));
            }
        }
        else if (data.attribute(m_attIndex).isString() && data.attribute(m_attIndex).name().startsWith("SEQ_")) {
            if (index == 1) {
                text.append(" contains ").append(m_shapeletPatternValue);
            }
            else {
                text.append(" !contains ").append(m_shapeletPatternValue);
            }
        }
        else if (data.attribute(m_attIndex).isString() && data.attribute(m_attIndex).name().startsWith("TS_")) {
            if (index == 0) {
                text.append(m_shapeletPatternValue).append(") <= ").append(m_splitPoint);
            }
            else {
                text.append(m_shapeletPatternValue).append(") > ").append(m_splitPoint);
            }
        }
        else if (data.attribute(m_attIndex).isString()) {
            System.err.println("Unexpected kind of attribute");
            System.exit(1);
        }
        else if (index == 0) {
            text.append(" <= ").append(m_splitPoint);
        }
        else {
            text.append(" > ").append(m_splitPoint);
        }
        
        return text.toString();
    }
    
    /**
     * Returns a string containing java source code equivalent to the test
     * made at this node. The instance being tested is called "i".
     *
     * @param index index of the nominal value tested
     * @param data  the data containing instance structure info
     *
     * @return a value of type 'String'
     */
    public final String sourceExpression(int index, Instances data) {
        StringBuilder expr = null;
        if (index < 0) {
            return "i[" + m_attIndex + "] == null";
        }
        else {
            if (data.attribute(m_attIndex).isNominal()) {
                if (index == 0) {
                    expr = new StringBuilder("i[");
                }
                else {
                    expr = new StringBuilder("!i[");
                }
                
                expr.append(m_attIndex).append("]");
                expr.append(".equals(\"").append(data.attribute(m_attIndex).value((int)m_splitPoint)).append("\")");
            }
            else if (data.attribute(m_attIndex).isString() && data.attribute(m_attIndex).name().startsWith("SEQ_")) {
                expr = new StringBuilder("Pattern.compile(");
                expr.append("\"(^|,|>)");
                expr.append("i[");
                expr.append(m_attIndex).append("]");
                expr.append(".replace(\",\", \"(,([a-zA-Z0-9+*-]+,)*)\")");
                if (m_maxGap > 0) {
                    expr.append(m_attIndex).append(".replace(\">\", \"(,[a-zA-Z0-9+*-]+)*>([a-zA-Z0-9,+*-]+>){0,")
                            .append(m_maxGap - 1).append("}").append("([a-zA-Z0-9+*-]+,)*\")");
                }
                else {
                    expr.append(m_attIndex)
                            .append(".replace(\">\", \"(,[a-zA-Z0-9+*-]+)*>([a-zA-Z0-9,+*-]+>)*([a-zA-Z0-9+*-]+,)*\")");
                }
                
                expr.append("(,|>|$)\"");
                expr.append(")");
                expr.append(".matcher(").append(data.attribute(m_attIndex).value(index)).append(")");
                expr.append(".find()");
                if (index == 0) {
                    expr.append(" == false");
                }
                else {
                    expr.append(" == true");
                }
            }
            else if (! data.attribute(m_attIndex).isString() || ! data.attribute(m_attIndex).name()
                    .startsWith("TS_")) {
                if (data.attribute(m_attIndex).isString()) {
                    System.err.println("Unexpected kind of attribute");
                    System.exit(1);
                }
                else {
                    expr = new StringBuilder("((Double) i[");
                    expr.append(m_attIndex).append("])");
                    if (index == 0) {
                        expr.append(".doubleValue() <= ").append(m_splitPoint);
                    }
                    else {
                        expr.append(".doubleValue() > ").append(m_splitPoint);
                    }
                }
            }
            assert expr != null;
            return expr.toString();
        }
    }
    
    /**
     * Sets split point to greatest value in given data smaller or equal to old split point.
     * (C4.5 does this for some strange reason).
     *
     * @param allInstances data to find split point within
     */
    public final void setSplitPoint(Instances allInstances) {
        if (allInstances.attribute(m_attIndex).isNumeric() && m_numSubsets > 1) {
            double newSplitPoint = - Double.MAX_VALUE;
            for (Instance inst : allInstances) {
                if (! inst.isMissing(m_attIndex)) {
                    double tempValue = inst.value(m_attIndex);
                    if (Utils.gr(tempValue, newSplitPoint) && Utils.smOrEq(tempValue, m_splitPoint)) {
                        newSplitPoint = tempValue;
                    }
                }
            }
            m_splitPoint = newSplitPoint;
        }
    }
    
    /**
     * Sets distribution associated with model.
     */
    public void resetDistribution(Instances data) throws Exception {
        Instances insts = new Instances(data, data.numInstances());
        
        for (int i = 0; i < data.numInstances(); i++) {
            if (this.whichSubset(data.instance(i)) > - 1) {
                insts.add(data.instance(i));
            }
        }
        
        Distribution newD = new Distribution(insts, this);
        newD.addInstWithUnknown(data, m_attIndex);
        m_distribution = newD;
    }
    
    /**
     * Returns weights if instance is assigned to more than one subset.
     * Returns null if instance is only assigned to one subset.
     */
    public final double[] weights(Instance instance) {
        if (! instance.isMissing(m_attIndex)) {
            return null;
        }
        
        double[] weights = new double[m_numSubsets];
        
        for (int i = 0; i < m_numSubsets; i++) {
            weights[i] = m_distribution.perBag(i) / m_distribution.total();
        }
        return weights;
    }
    
    /**
     * Returns index of subset instance is assigned to. Returns -1 if instance is assigned to more than one subset.
     */
    public final int whichSubset(Instance instance) {
        if (instance.isMissing(m_attIndex)) {
            return - 1;
        }
        
        if (instance.attribute(m_attIndex).isNominal()) {
            return (int)instance.value(m_attIndex);
        }
        
        if (instance.attribute(m_attIndex).isString() && instance.attribute(m_attIndex).name().startsWith("SEQ_")) {
            String sequence      = instance.stringValue(m_attIndex);
            String pattern_regex = m_shapeletPatternValue.replace(",", "(,([a-zA-Z0-9+*-]+,)*)");
            if (m_maxGap > 0) {
                pattern_regex = pattern_regex.replace(">",
                        "(,[a-zA-Z0-9+*-]+)*>([a-zA-Z0-9,+*-]+>){0," + (m_maxGap - 1) + "}" + "([a-zA-Z0-9" +
                                "+*-]+,)*");
            }
            else {
                pattern_regex = pattern_regex.replace(">", "(,[a-zA-Z0-9+*-]+)*>([a-zA-Z0-9,+*-]+>)*" +
                        "([a-zA-Z0-9+*-]+,)*");
            }
            
            pattern_regex = "(^|,|>)" + pattern_regex + "(,|>|$)";
            Pattern p     = Pattern.compile(pattern_regex);
            Matcher m     = p.matcher(sequence);
            boolean found = m.find();
            return found ? 1 : 0;
        }
        
        if (instance.attribute(m_attIndex).isString() && instance.attribute(m_attIndex).name().startsWith("TS_")) {
            String   sequence      = instance.stringValue(m_attIndex);
            String[] tsSplitString = sequence.split(",");
            double[] tsDouble      = new double[tsSplitString.length];
            
            for (int i = 0; i < tsSplitString.length; i++) {
                tsDouble[i] = Double.parseDouble(tsSplitString[i]);
            }
            
            double distance = subsequenceDistOpt(tsDouble, m_shapeletArray);
            return Utils.smOrEq(distance, m_splitPoint) ? 0 : 1;
        }
        
        if (instance.attribute(m_attIndex).isString()) {
            System.err.println("Unexpected kind of attribute");
            System.exit(1);
        }
        return Utils.smOrEq(instance.value(m_attIndex), m_splitPoint) ? 0 : 1;
    }
    
    /**
     * Returns the revision string.
     *
     * @return the revision
     */
    public String getRevision() { return RevisionUtils.extract("$Revision: 10535 $"); }
    
    public double DTW(double[] timeSeries, int baseIndex, double[] shapelet, double bestValue) {
        double[][] distanceMatrix = new double[shapelet.length][shapelet.length];
        
        for (int i = 0; i < shapelet.length; i++) {
            distanceMatrix[0][i] = Double.MAX_VALUE;
            distanceMatrix[i][0] = Double.MAX_VALUE;
            distanceMatrix[0][0] = 0.0D;
        }
        
        for (int i = 1; i < shapelet.length; i++) {
            for (int j = 1; j < shapelet.length; j++) {
                double cellCost = Math.pow(timeSeries[baseIndex + i] - shapelet[j], 2.0D);
                double minValue;
                double a        = distanceMatrix[i - 1][j];
                double b        = distanceMatrix[i][j - 1];
                double c        = distanceMatrix[i - 1][j - 1];
                if (a < b && a < c) {
                    minValue = a;
                }
                else if (b < c && b < a) {
                    minValue = b;
                }
                else {
                    minValue = c;
                }
                
                distanceMatrix[i][j] = cellCost + minValue;
                if (distanceMatrix[i][j] > bestValue) {
                    return distanceMatrix[i][j];
                }
            }
        }
        return Math.sqrt(distanceMatrix[shapelet.length - 1][shapelet.length - 1]);
    }
    
    private double subsequenceDistOpt(double[] timeSeries, double[] shapelet) {
        double minimumDistance = Double.MAX_VALUE;
        if (shapelet.length > timeSeries.length) {
            double[] temp = timeSeries;
            timeSeries = shapelet;
            shapelet   = temp;
        }
        
        for (int i = 0; i < timeSeries.length - shapelet.length + 1; i++) {
            boolean stop        = false;
            double  sumDistance = 0.0D;
            
            for (int j = 0; j < shapelet.length; j++) {
                double tsValue = timeSeries[i + j];
                sumDistance += Math.pow(shapelet[j] - tsValue, 2.0D);
                if (sumDistance >= minimumDistance) {
                    stop = true;
                    break;
                }
            }
            
            if (! stop) {
                minimumDistance = sumDistance;
            }
        }
        
        double sqrtVal = Math.sqrt(minimumDistance);
        return round(sqrtVal, 5);
    }
    
    public static double round(double value, int places) {
        double newval = value * Math.pow(10.0D, places);
        newval = ((int)newval);
        newval /= Math.pow(10.0D, places);
        return newval;
    }
}
