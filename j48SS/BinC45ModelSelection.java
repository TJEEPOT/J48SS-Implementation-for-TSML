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

import tsml.data_containers.TimeSeriesInstances;
import weka.core.Instances;
import weka.core.RevisionUtils;
import weka.core.Utils;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Class for selecting a C4.5-like binary (!) split for a given dataset.
 *
 * @author Eibe Frank (eibe@cs.waikato.ac.nz)
 * @version $Revision: 10539 $
 */
public class BinC45ModelSelection extends ModelSelection {
    /** for serialization */
    private static final long serialVersionUID = 1473770058126067587L;
    
    /** Minimum number of instances that have to occur in at least two subsets induced by split. */
    protected final int m_minNoObj;
    
    /** Use MDL correction when finding splits on numeric attributes? */
    protected final boolean m_useMDLcorrection;
    
    /** The training dataset from .ARFF. */
    protected Instances m_instancesData;
    
    /** The training dataset from .TS. */
    protected TimeSeriesInstances m_tsData = null;
    
    /** The minimum support that the extracted sequential patterns must have (between 0 and 1) */
    protected double m_minimumSupport;
    
    /** If true, uses IG scoring of the best non-sequential attribute to guide the pruning of pattern search space */
    protected boolean m_useIGPruning;
    
    /** The maximum gap between two itemsets in a pattern (1 = No gap) */
    protected int m_maxGap;
    
    /** The maximum pattern length in terms of itemset count */
    protected int m_maxPatternLength;
    
    /** The weight used to evaluate the patterns */
    protected double m_patternWeight;
    
    /** The population size of the genetic algorithm i.e., the number of individuals */
    private final int m_popSize;
    
    /**
     * The number of evaluations that are going to be carried out in the optimization process (should be higher than
     * popSize)
     */
    private final int m_numEvals;
    
    /** The probability of combining two individuals of the population */
    private final double m_crossoverProb;
    
    /** The  probability of an element to undergo a random mutation */
    private final double m_mutationProb;
    
    private Map<String, String> m_toVGEN   = new HashMap<>();
    private Map<String, String> m_fromVGEN = new HashMap<>();
    
    protected final boolean m_doNotMakeSplitPointActualValue;
    
    /** Should the classifier print out what it's doing to console? */
    protected boolean m_isVerbose;
    
    /**
     * Constructor for Instances (.ARFF) data.
     */
    public BinC45ModelSelection(
            int minNoObj, Instances allData, boolean useMDLcorrection, boolean doNotMakeSplitPointActualValue,
            double minimumSupport, boolean useIGPruning, int maxGap, int maxPatLength, double patternWeight,
            int popSize, int numEvals, double crossoverProb, double mutationP, boolean isVerbose) {
        m_minNoObj                       = minNoObj;
        m_instancesData                  = allData;
        m_useMDLcorrection               = useMDLcorrection;
        m_doNotMakeSplitPointActualValue = doNotMakeSplitPointActualValue;
        m_minimumSupport                 = minimumSupport;
        m_useIGPruning                   = useIGPruning;
        m_maxGap                         = maxGap;
        m_maxPatternLength               = maxPatLength;
        m_patternWeight                  = patternWeight;
        m_popSize                        = popSize;
        m_numEvals                       = numEvals;
        m_crossoverProb                  = crossoverProb;
        m_mutationProb                   = mutationP;
        m_isVerbose                      = isVerbose;
    }
    
    /**
     * Constructor for TimeSeriesInstances (.TS) data.
     */
    public BinC45ModelSelection(
            int minNoObj, TimeSeriesInstances allData, boolean useMDLcorrection, boolean doNotMakeSplitPointActualValue,
            double minimumSupport, boolean useIGPruning, int maxGap, int maxPatLength, double patternWeight,
            int popSize, int numEvals, double crossoverP, double mutationP, boolean isVerbose) {
        m_minNoObj                       = minNoObj;
        m_instancesData                  = null;
        m_tsData                         = allData;
        m_useMDLcorrection               = useMDLcorrection;
        m_doNotMakeSplitPointActualValue = doNotMakeSplitPointActualValue;
        m_minimumSupport                 = minimumSupport;
        m_useIGPruning                   = useIGPruning;
        m_maxGap                         = maxGap;
        m_maxPatternLength               = maxPatLength;
        m_patternWeight                  = patternWeight;
        m_popSize                        = popSize;
        m_numEvals                       = numEvals;
        m_crossoverProb                  = crossoverP;
        m_mutationProb                   = mutationP;
        m_isVerbose                      = isVerbose;
    }
    
    public void setItemTranslationToVGEN(Map<String, String> toVGEN)     { m_toVGEN = toVGEN; }
    
    public void setItemTranslationFromVGEN(Map<String, String> fromVGEN) {m_fromVGEN = fromVGEN;}
    
    /**
     * Sets reference to training data to null.
     */
    public void cleanup() {
        m_instancesData = null;
        m_tsData        = null;
    }
    
    /**
     * Selects C4.5-type split for the given dataset.
     */
    public final ClassifierSplitModel selectModel(Instances data) throws Exception {
        NoSplit noSplitModel;
        
        // Check if all Instances belong to one class or if not enough Instances to split.
        Distribution checkDistribution = new Distribution(data);
        noSplitModel = new NoSplit(checkDistribution);
        if (Utils.sm(checkDistribution.total(), (2 * m_minNoObj)) || Utils.eq(checkDistribution.total(),
                checkDistribution.perClass(checkDistribution.maxClass()))) {
            return noSplitModel;
        }
        
        BinC45Split[] currentModel         = new BinC45Split[data.numAttributes()];
        double        sumOfWeights         = data.sumOfWeights();
        List<Integer> sequentialAttIndices = new ArrayList<>();
        List<Integer> tsAttIndices         = new ArrayList<>();
        
        // Check what type of data we're dealing with
        for (int i = 0; i < data.numAttributes(); i++) {
            if (data.attribute(i).isString() && data.attribute(i).name().startsWith("SEQ_")) {
                sequentialAttIndices.add(i);
            }
            else if (data.attribute(i).isString() && data.attribute(i).name().startsWith("TS_")) {
                tsAttIndices.add(i);
            }
            else if (i != data.classIndex()) {
                currentModel[i] = new BinC45Split(i, m_minNoObj, sumOfWeights, m_useMDLcorrection);
                currentModel[i].buildClassifier(data);
                currentModel[i].checkModel();
            }
            else {
                currentModel[i] = null;
            }
        }
        
        // Find "best" attribute to split on.
        double      minResult = 0.0D;
        BinC45Split bestModel = null;
        
        // build the best model in the numeric models
        for (int i = 0; i < data.numAttributes(); i++) {
            if (i != data.classIndex() && ! data.attribute(i).isString()) {
                assert currentModel[i] != null;
                if (currentModel[i].checkModel() && Utils.gr(currentModel[i].infoGain(), minResult)) {
                    bestModel = currentModel[i];
                    minResult = currentModel[i].infoGain();
                }
            }
        }
        
        double[] minResultOneVsAll = new double[m_instancesData.numClasses()];
        for (int i = 0; i < minResultOneVsAll.length; i++) {
            if (bestModel == null) {
                minResultOneVsAll[i] = 0.0D;
            }
            else if (m_useIGPruning) {
                minResultOneVsAll[i] = BinC45Split.m_infoGainCrit.splitCritValueOneVsAll(bestModel.m_distribution,
                        bestModel.m_sumOfWeights, i); ;
            }
            else {
                minResultOneVsAll[i] = 0.0D;
            }
        }
        
        // build the sequential models
        for (int index : sequentialAttIndices) {
            currentModel[index] = new BinC45Split(index, m_minNoObj, sumOfWeights, m_useMDLcorrection,
                    minResultOneVsAll, m_minimumSupport, m_toVGEN, m_fromVGEN, m_maxGap, m_maxPatternLength,
                    m_patternWeight, m_isVerbose);
            currentModel[index].buildClassifier(data);
            if (currentModel[index].checkModel()) {
                if (Utils.gr(currentModel[index].infoGain(), minResult)) {
                    bestModel = currentModel[index];
                    minResult = currentModel[index].infoGain();
                }
            }
        }
        
        // build the time-series models
        for (int index : tsAttIndices) {
            currentModel[index] = new BinC45Split(index, m_minNoObj, sumOfWeights, m_useMDLcorrection, m_popSize,
                    m_numEvals, m_crossoverProb, m_mutationProb, m_patternWeight, m_isVerbose);
            currentModel[index].buildClassifier(data);
            if (currentModel[index].checkModel()) {
                if (Utils.gr(currentModel[index].infoGain(), minResult)) {
                    bestModel = currentModel[index];
                    minResult = currentModel[index].infoGain();
                }
            }
        }
        
        // Check if useful split was found.
        if (Utils.eq(minResult, 0.0D)) {
            return noSplitModel;
        }
        
        // Add all Instances with unknown values for the corresponding attribute to the distribution for the model, so
        // that the complete distribution is stored with the model.
        assert bestModel != null;
        bestModel.distribution().addInstWithUnknown(data, bestModel.attIndex());
        if (m_instancesData != null && ! m_doNotMakeSplitPointActualValue) {
            // Set the split point analogue to C45 if attribute numeric.
            bestModel.setSplitPoint(m_instancesData);
        }
        return bestModel;
    }
    
    /**
     * Selects C4.5-type split for the given dataset.
     */
    public final ClassifierSplitModel selectModel(Instances train, Instances test) throws Exception {
        return this.selectModel(train);
    }
    
    /**
     * Returns the revision string.
     *
     * @return the revision
     */
    public String getRevision() { return RevisionUtils.extract("$Revision: 10540 $"); }
}
