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

import weka.core.Instance;
import weka.core.Instances;
import weka.core.RevisionHandler;
import weka.core.Utils;

import java.io.Serializable;

/**
 * Abstract class for classification models that can be used
 * recursively to split the data.
 *
 * @author Eibe Frank (eibe@cs.waikato.ac.nz)
 * @version $Revision: 8034 $
 */
public abstract class ClassifierSplitModel implements Cloneable, Serializable, RevisionHandler {
    /** for serialization */
    private static final long serialVersionUID = 6665593021852893129L;
    
    /** Distribution of class values. */
    protected Distribution m_distribution;
    
    /** Number of created subsets. */
    protected int m_numSubsets;
    
    public ClassifierSplitModel() {}
    
    /**
     * Allows to clone a model (shallow copy).
     */
    public Object clone() throws CloneNotSupportedException { return super.clone(); }
    
    /**
     * Builds the classifier split model for the given set of instances.
     *
     * @param instances Instances to be used to build the model
     *
     * @throws Exception if something goes wrong
     */
    public abstract void buildClassifier(Instances instances) throws Exception;
    
    /**
     * Checks if generated model is valid.
     *
     * @return True if model is valid
     */
    public final boolean checkModel() { return m_numSubsets > 0; }
    
    /**
     * Gets class probability for instance.
     *
     * @param classIndex index to use for checking data
     * @param instance   the data
     * @param theSubset  subset for comparing to the data
     *
     * @return class probability for the data
     */
    public double classProb(int classIndex, Instance instance, int theSubset) {
        if (theSubset > - 1) {
            return m_distribution.prob(classIndex, theSubset);
        }
        
        double[] weights = this.weights(instance);
        if (weights == null) {
            return m_distribution.prob(classIndex);
        }
        
        double prob = 0.0D;
        for (int i = 0; i < weights.length; i++) {
            prob += weights[i] * m_distribution.prob(classIndex, i);
        }
        return prob;
    }
    
    /**
     * Gets class probability for instance.
     *
     * @param classIndex index to use for checking data
     * @param instance   the data
     * @param theSubset  subset for comparing to the data
     *
     * @return class probability for the data
     */
    public double classProbLaplace(int classIndex, Instance instance, int theSubset) {
        if (theSubset > - 1) {
            return m_distribution.laplaceProb(classIndex, theSubset);
        }
        
        double[] weights = this.weights(instance);
        if (weights == null) {
            return m_distribution.laplaceProb(classIndex);
        }
        
        double prob = 0.0D;
        for (int i = 0; i < weights.length; i++) {
            prob += weights[i] * m_distribution.laplaceProb(classIndex, i);
        }
        return prob;
        
    }
    
    /**
     * @return the distribution of class values induced by the model.
     */
    public final Distribution distribution() { return m_distribution; }
    
    /**
     * Prints the left side of data.
     *
     * @param data the data.
     *
     * @return left side of condition satisfied by data.
     */
    public abstract String leftSide(Instances data);
    
    /**
     * Prints the right side of data.
     *
     * @param index The subset index
     * @param data  The data within the subset
     *
     * @return the left side of condition satisfied by instances in subset index
     */
    public abstract String rightSide(int index, Instances data);
    
    /**
     * Prints the dump label.
     *
     * @param index subset index of item that needs to be dumped
     * @param data  data to be sorted
     *
     * @return the label for subset index of instances (eg class)
     */
    public final String dumpLabel(int index, Instances data) {
        StringBuilder text = new StringBuilder();
        text.append(data.classAttribute().value(m_distribution.maxClass(index)));
        text.append(" (").append(Utils.roundDouble(m_distribution.perBag(index), 2));
        
        if (Utils.gr(m_distribution.numIncorrect(index), 0.0D)) {
            text.append("/").append(Utils.roundDouble(m_distribution.numIncorrect(index), 2));
        }
        text.append(")");
        return text.toString();
    }
    
    public abstract String sourceExpression(int index, Instances data);
    
    /**
     * @return the number of created subsets for the split.
     */
    public final int numSubsets() { return m_numSubsets; }
    
    /**
     * Sets distribution associated with model.
     */
    public void resetDistribution(Instances data) throws Exception { m_distribution = new Distribution(data, this); }
    
    /**
     * Splits the given set of instances into subsets.
     *
     * @throws Exception if something goes wrong
     */
    public final Instances[] split(Instances data) throws Exception {
        int[] subsetSize = new int[m_numSubsets];
        
        for (Instance instance : data) {
            int subset = this.whichSubset(instance);
            if (subset > - 1) {
                subsetSize[subset] += 1;
            }
            else {
                double[] weights = this.weights(instance);
                
                for (int j = 0; j < m_numSubsets; j++) {
                    if (Utils.gr(weights[j], 0.0D)) {
                        subsetSize[j] += 1;
                    }
                }
            }
        }
        
        Instances[] splitInstances = new Instances[m_numSubsets];
        
        for (int i = 0; i < m_numSubsets; i++) {
            splitInstances[i] = new Instances(data, subsetSize[i]);
        }
        
        for (Instance instance : data) {
            int subset = this.whichSubset(instance);
            if (subset > - 1) {
                splitInstances[subset].add(instance);
            }
            else {
                double[] weights = this.weights(instance);
                
                for (int j = 0; j < m_numSubsets; j++) {
                    if (Utils.gr(weights[j], 0.0D)) {
                        splitInstances[j].add(instance);
                        splitInstances[j].lastInstance().setWeight(weights[j] * instance.weight());
                    }
                }
            }
        }
        return splitInstances;
    }
    
    /**
     * Returns null if instance is only assigned to one subset.
     *
     * @param instance instance to look for weights
     *
     * @return weights if instance is assigned to more than one subset.
     */
    public abstract double[] weights(Instance instance);
    
    /**
     * Returns -1 if instance is assigned to more than one subset.
     *
     * @param instance instance to look for subset
     *
     * @return index of subset instance is assigned to.
     */
    public abstract int whichSubset(Instance instance);
}
