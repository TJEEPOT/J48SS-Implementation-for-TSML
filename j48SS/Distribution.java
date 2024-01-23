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

import weka.core.*;

import java.io.Serializable;
import java.util.Arrays;

/**
 * Class for handling a distribution of class values.
 *
 * @author Eibe Frank (eibe@cs.waikato.ac.nz)
 * @version $Revision: 8034 $
 */
public class Distribution implements Cloneable, Serializable, RevisionHandler {
    /** for serialization */
    private static final long serialVersionUID = 8526859638230806576L;
    
    /** Weight of instances per class per bag. */
    protected final double[][] m_perClassPerBag;
    
    /** Weight of instances per bag. */
    protected final double[] m_perBag;
    
    /** Weight of instances per class. */
    protected final double[] m_perClass;
    
    /** Total weight of instances. */
    protected double m_totalWeight;
    
    /**
     * Creates and initializes a new distribution.
     */
    public Distribution(int numBags, int numClasses) {
        m_perClassPerBag = new double[numBags][0];
        m_perBag         = new double[numBags];
        m_perClass       = new double[numClasses];
        
        for (int i = 0; i < numBags; i++) {
            m_perClassPerBag[i] = new double[numClasses];
        }
        m_totalWeight = 0.0D;
    }
    
    /**
     * Creates and initializes a new distribution using the given array.
     * WARNING: it just copies a reference to this array.
     */
    public Distribution(double[][] table) {
        m_perClassPerBag = table;
        m_perBag         = new double[table.length];
        m_perClass       = new double[table[0].length];
        
        for (int i = 0; i < table.length; i++) {
            for (int j = 0; j < table[i].length; j++) {
                m_perBag[i] += table[i][j];
                m_perClass[j] += table[i][j];
                m_totalWeight += table[i][j];
            }
        }
    }
    
    /**
     * Creates a distribution with only one bag according to instances in source.
     */
    public Distribution(Instances data) {
        m_perClassPerBag    = new double[1][0];
        m_perBag            = new double[1];
        m_totalWeight       = 0.0D;
        m_perClass          = new double[data.numClasses()];
        m_perClassPerBag[0] = new double[data.numClasses()];
        
        for (Instance inst : data) {
            this.add(0, inst);
        }
    }
    
    /**
     * Creates a distribution according to given instances and
     * split model.
     */
    public Distribution(Instances data, ClassifierSplitModel modelToUse) {
        m_perClassPerBag = new double[modelToUse.numSubsets()][0];
        m_perBag         = new double[modelToUse.numSubsets()];
        m_totalWeight    = 0.0D;
        m_perClass       = new double[data.numClasses()];
        
        for (int i = 0; i < modelToUse.numSubsets(); i++) {
            m_perClassPerBag[i] = new double[data.numClasses()];
        }
        
        for (Instance instance : data) {
            int index = modelToUse.whichSubset(instance);
            if (index != - 1) {
                this.add(index, instance);
            }
            else {
                double[] weights = modelToUse.weights(instance);
                this.addWeights(instance, weights);
            }
        }
    }
    
    /**
     * Creates distribution with only one bag by merging all bags of given distribution.
     */
    public Distribution(Distribution toMerge) {
        m_totalWeight = toMerge.m_totalWeight;
        m_perClass    = new double[toMerge.numClasses()];
        System.arraycopy(toMerge.m_perClass, 0, m_perClass, 0, toMerge.numClasses());
        
        m_perClassPerBag    = new double[1][0];
        m_perClassPerBag[0] = new double[toMerge.numClasses()];
        System.arraycopy(toMerge.m_perClass, 0, m_perClassPerBag[0], 0, toMerge.numClasses());
        
        m_perBag    = new double[1];
        m_perBag[0] = m_totalWeight;
    }
    
    /**
     * Creates distribution with two bags by merging all bags apart of the indicated one.
     */
    public Distribution(Distribution toMerge, int index) {
        m_totalWeight = toMerge.m_totalWeight;
        m_perClass    = new double[toMerge.numClasses()];
        System.arraycopy(toMerge.m_perClass, 0, m_perClass, 0, toMerge.numClasses());
        m_perClassPerBag    = new double[2][0];
        m_perClassPerBag[0] = new double[toMerge.numClasses()];
        System.arraycopy(toMerge.m_perClassPerBag[index], 0, m_perClassPerBag[0], 0, toMerge.numClasses());
        m_perClassPerBag[1] = new double[toMerge.numClasses()];
        
        for (int i = 0; i < toMerge.numClasses(); i++) {
            m_perClassPerBag[1][i] = toMerge.m_perClass[i] - m_perClassPerBag[0][i];
        }
        
        m_perBag    = new double[2];
        m_perBag[0] = toMerge.m_perBag[index];
        m_perBag[1] = m_totalWeight - m_perBag[0];
    }
    
    /**
     * Subtracts instance from given bag.
     */
    public final void sub(int bagIndex, Instance instance) {
        int    classIndex = (int)instance.classValue();
        double weight     = instance.weight();
        m_perClassPerBag[bagIndex][classIndex] -= weight;
        m_perBag[bagIndex] -=                     weight;
        m_perClass[classIndex] -=                 weight;
        
        m_totalWeight -= weight;
    }
    
    /**
     * Adds instance to given bag.
     */
    public final void add(int bagIndex, Instance instance) {
        int    classIndex = (int)instance.classValue();
        double weight     = instance.weight();
        m_perClassPerBag[bagIndex][classIndex] += weight;
        m_perBag[bagIndex] +=                     weight;
        m_perClass[classIndex] +=                 weight;
        
        m_totalWeight += weight;
    }
    
    /**
     * Adds counts to given bag.
     */
    public final void add(int bagIndex, double[] counts) {
        double sum = Utils.sum(counts);
        
        for (int i = 0; i < counts.length; i++) {
            m_perClassPerBag[bagIndex][i] += counts[i];
        }
        m_perBag[bagIndex] += sum;
        
        for (int i = 0; i < counts.length; i++) {
            m_perClass[i] += counts[i];
        }
        m_totalWeight += sum;
    }
    
    /**
     * Adds all instances with unknown values for given attribute, weighted according to frequency of instances in
     * each bag.
     */
    public final void addInstWithUnknown(Instances data, int attIndex) {
        double[] probs = new double[m_perBag.length];
        
        for (int j = 0; j < m_perBag.length; j++) {
            if (Utils.eq(m_totalWeight, 0.0D)) {
                probs[j] = 1.0D / (double)probs.length;
            }
            else {
                probs[j] = m_perBag[j] / m_totalWeight;
            }
        }
        
        for (Instance instance : data) {
            if (instance.isMissing(attIndex)) {
                int    classIndex = (int)instance.classValue();
                double weight     = instance.weight();
                m_perClass[classIndex] += weight;
                                          m_totalWeight += weight;
                
                for (int i = 0; i < m_perBag.length; i++) {
                    double newWeight = probs[i] * weight;
                    m_perClassPerBag[i][classIndex] += newWeight;
                    m_perBag[i] +=                     newWeight;
                }
            }
        }
    }
    
    /**
     * Adds all instances in given range to given bag.
     */
    public final void addRange(int bagIndex, Instances data, int startIndex, int lastPlusOne) {
        double sumOfWeights = 0.0D;
        
        for (int i = startIndex; i < lastPlusOne; i++) {
            Instance instance   = data.instance(i);
            int      classIndex = (int)instance.classValue();
            sumOfWeights += instance.weight();
            m_perClassPerBag[bagIndex][classIndex] += instance.weight();
            m_perClass[classIndex] += instance.weight();
        }
        m_perBag[bagIndex] += sumOfWeights;
        
        m_totalWeight += sumOfWeights;
    }
    
    /**
     * Adds given instance to all bags weighting it according to given weights.
     */
    public final void addWeights(Instance instance, double[] weights) {
        int classIndex = (int)instance.classValue();
        
        for (int i = 0; i < m_perBag.length; i++) {
            double weight = instance.weight() * weights[i];
            m_perClassPerBag[i][classIndex] += weight;
            m_perBag[i] +=                     weight;
            m_perClass[classIndex] +=          weight;
            
            m_totalWeight += weight;
        }
    }
    
    /**
     * Checks if at least two bags contain a minimum number of instances.
     */
    public final boolean check(double minNoObj) {
        int counter = 0;
        
        for (double v : m_perBag) {
            if (Utils.grOrEq(v, minNoObj)) {
                counter++;
            }
        }
        return counter > 1;
    }
    
    /**
     * Clones distribution (Deep copy of distribution).
     */
    public final Object clone() {
        Distribution newDistribution = new Distribution(m_perBag.length, m_perClass.length);
        
        for (int i = 0; i < m_perBag.length; i++) {
            newDistribution.m_perBag[i] = m_perBag[i];
            System.arraycopy(m_perClassPerBag[i], 0, newDistribution.m_perClassPerBag[i], 0, m_perClass.length);
        }
        System.arraycopy(m_perClass, 0, newDistribution.m_perClass, 0, m_perClass.length);
        newDistribution.m_totalWeight = m_totalWeight;
        return newDistribution;
    }
    
    /**
     * Deletes given instance from given bag.
     */
    public final void del(int bagIndex, Instance instance) {
        int    classIndex = (int)instance.classValue();
        double weight     = instance.weight();
        
        m_perClassPerBag[bagIndex][classIndex] -= weight;
        m_perBag[bagIndex] -=                     weight;
        m_perClass[classIndex] -=                 weight;
        
        m_totalWeight -= weight;
    }
    
    /**
     * Sets all counts to zero.
     */
    public final void initialize() {
        Arrays.fill(m_perClass, 0.0D);
        Arrays.fill(m_perBag, 0.0D);
        
        for (int i = 0; i < m_perBag.length; i++) {
            for (int j = 0; j < m_perClass.length; j++) {
                m_perClassPerBag[i][j] = 0.0D;
            }
        }
        m_totalWeight = 0.0D;
    }
    
    /**
     * Returns matrix with distribution of class values.
     */
    public final double[][] matrix() { return m_perClassPerBag; }
    
    /**
     * Returns index of bag containing maximum number of instances.
     */
    public final int maxBag() {
        double max      = 0.0D;
        int    maxIndex = - 1;
        
        for (int i = 0; i < m_perBag.length; i++) {
            if (Utils.grOrEq(m_perBag[i], max)) {
                max      = m_perBag[i];
                maxIndex = i;
            }
        }
        return maxIndex;
    }
    
    /**
     * Returns class with highest frequency over all bags.
     */
    public final int maxClass() {
        double maxCount = 0.0D;
        int    maxIndex = 0;
        
        for (int i = 0; i < m_perClass.length; i++) {
            if (Utils.gr(m_perClass[i], maxCount)) {
                maxCount = m_perClass[i];
                maxIndex = i;
            }
        }
        return maxIndex;
    }
    
    /**
     * Returns class with highest frequency for given bag.
     */
    public final int maxClass(int index) {
        double maxCount = 0.0D;
        int    maxIndex = 0;
        
        if (Utils.gr(m_perBag[index], 0.0D)) {
            for (int i = 0; i < m_perClass.length; i++) {
                if (Utils.gr(m_perClassPerBag[index][i], maxCount)) {
                    maxCount = m_perClassPerBag[index][i];
                    maxIndex = i;
                }
            }
            return maxIndex;
        }
        return this.maxClass();
    }
    
    /**
     * Returns number of bags.
     */
    public final int numBags() { return m_perBag.length; }
    
    /**
     * Returns number of classes.
     */
    public final int numClasses() { return m_perClass.length; }
    
    /**
     * Returns the max class for m_perClass.
     */
    public final double numCorrect() { return m_perClass[this.maxClass()]; }
    
    /**
     * Returns the max class for the given class bag.
     */
    public final double numCorrect(int index) { return m_perClassPerBag[index][this.maxClass(index)]; }
    
    /**
     * Returns total of the weights of classes that are not the max.
     */
    public final double numIncorrect() { return m_totalWeight - this.numCorrect(); }
    
    /**
     * Returns the total of the weights of classes from the given bag that are not the max .
     */
    public final double numIncorrect(int index) {
        return m_perBag[index] - this.numCorrect(index);
    }
    
    /**
     * Returns number of (possibly fractional) instances of given class in given bag.
     */
    public final double perClassPerBag(int bagIndex, int classIndex) { return m_perClassPerBag[bagIndex][classIndex]; }
    
    /**
     * Returns number of (possibly fractional) instances in given bag.
     */
    public final double perBag(int bagIndex) { return m_perBag[bagIndex]; }
    
    /**
     * Returns number of (possibly fractional) instances of given class.
     */
    public final double perClass(int classIndex) { return m_perClass[classIndex]; }
    
    /**
     * Returns relative frequency of class over all bags with Laplace correction.
     */
    public final double laplaceProb(int classIndex) {
        return (m_perClass[classIndex] + 1.0D) / (m_totalWeight + (double)m_perClass.length);
    }
    
    /**
     * Returns relative frequency of class for given bag.
     */
    public final double laplaceProb(int classIndex, int intIndex) {
        if (Utils.gr(m_perBag[intIndex], 0.0D)) {
            return (m_perClassPerBag[intIndex][classIndex] + 1.0D) / (m_perBag[intIndex] + (double)m_perClass.length);
        }
        return this.laplaceProb(classIndex);
    }
    
    /**
     * Returns relative frequency of class over all bags.
     */
    public final double prob(int classIndex) {
        if (! Utils.eq(m_totalWeight, 0.0D)) {
            return m_perClass[classIndex] / m_totalWeight;
        }
        return 0.0D;
    }
    
    /**
     * Returns relative frequency of class for given bag.
     */
    public final double prob(int classIndex, int intIndex) {
        if (Utils.gr(m_perBag[intIndex], 0.0D)) {
            return m_perClassPerBag[intIndex][classIndex] / m_perBag[intIndex];
        }
        return this.prob(classIndex);
    }
    
    /**
     * Subtracts the given distribution from this one. The results has only one bag.
     */
    public final Distribution subtract(Distribution toSubstract) {
        Distribution newDist = new Distribution(1, m_perClass.length);
        newDist.m_perBag[0]   = m_totalWeight - toSubstract.m_totalWeight;
        newDist.m_totalWeight = newDist.m_perBag[0];
        
        for (int i = 0; i < m_perClass.length; i++) {
            newDist.m_perClassPerBag[0][i] = m_perClass[i] - toSubstract.m_perClass[i];
            newDist.m_perClass[i]          = newDist.m_perClassPerBag[0][i];
        }
        return newDist;
    }
    
    /**
     * Returns total number of (possibly fractional) instances.
     */
    public final double total() { return m_totalWeight; }
    
    /**
     * Shifts given instance from one bag to another one.
     */
    public final void shift(int from, int to, Instance instance) {
        int classIndex = (int)instance.classValue();
        m_perClassPerBag[from][classIndex] -= instance.weight();
        m_perClassPerBag[to][classIndex] += instance.weight();
        m_perBag[from] -= instance.weight();
        m_perBag[to] += instance.weight();
    }
    
    /**
     * Shifts all instances in given range from one bag to another one.
     */
    public final void shiftRange(int from, int to, Instances data, int startIndex, int lastPlusOne) {
        for (int i = startIndex; i < lastPlusOne; i++) {
            Instance instance   = data.instance(i);
            int      classIndex = (int)instance.classValue();
            m_perClassPerBag[from][classIndex] -= instance.weight();
            m_perClassPerBag[to][classIndex] += instance.weight();
            m_perBag[from] -= instance.weight();
            m_perBag[to] += instance.weight();
        }
    }
    
    /**
     * Returns the revision string.
     */
    public String getRevision() { return RevisionUtils.extract("$Revision: 10531 $"); }
}
