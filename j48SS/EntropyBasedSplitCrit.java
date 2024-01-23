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

import weka.core.RevisionHandler;

import java.io.Serializable;

/**
 * "Abstract" class for computing splitting criteria based on the entropy of a class distribution.
 *
 * @author Eibe Frank (eibe@cs.waikato.ac.nz)
 * @version $Revision: 8034 $
 */
public abstract class EntropyBasedSplitCrit implements Serializable, RevisionHandler {
    
    /** for serialization */
    private static final long serialVersionUID = - 2618691439791653056L;
    
    /**
     * Default constructor
     */
    public EntropyBasedSplitCrit() {}
    
    /**
     * Help method for computing entropy.
     */
    public final double lnFunc(double num) { return num < 1.0E-6D ? 0.0D : num * Math.log(num); }
    
    /**
     * Computes entropy of distribution before splitting.
     */
    public final double oldEnt(Distribution bags) {
        double returnValue = 0.0D;
        
        for (int j = 0; j < bags.numClasses(); j++) {
            returnValue += this.lnFunc(bags.perClass(j));
        }
        return (this.lnFunc(bags.total()) - returnValue) / Math.log(2);
    }
    
    /**
     * Computes entropy of distribution after splitting.
     */
    public final double newEnt(Distribution bags) {
        double returnValue = 0.0D;
        
        for (int i = 0; i < bags.numBags(); i++) {
            for (int j = 0; j < bags.numClasses(); j++) {
                returnValue += this.lnFunc(bags.perClassPerBag(i, j));
            }
            returnValue -= this.lnFunc(bags.perBag(i));
        }
        return - (returnValue / Math.log(2));
    }
    
    /**
     * Computes entropy of distribution before splitting.
     */
    public final double oldEnt(Distribution bags, int oneVsAllClassIndex) {
        double returnValue               = this.lnFunc(bags.perClass(oneVsAllClassIndex));
        double cumulativeNumberInstances = 0.0D;
        
        for (int j = 0; j < bags.numClasses(); j++) {
            if (j != oneVsAllClassIndex) {
                cumulativeNumberInstances += bags.perClass(j);
            }
        }
        returnValue += this.lnFunc(cumulativeNumberInstances);
        return (this.lnFunc(bags.total()) - returnValue) / Math.log(2);
    }
    
    /**
     * Computes entropy of distribution after splitting.
     */
    public final double newEnt(Distribution bags, int oneVsAllClassIndex) {
        double returnValue = 0.0D;
        
        for (int i = 0; i < bags.numBags(); i++) {
            returnValue += this.lnFunc(bags.perClassPerBag(i, oneVsAllClassIndex));
            double cumulativeNumberInstances = 0.0D;
            
            for (int j = 0; j < bags.numClasses(); j++) {
                if (j != oneVsAllClassIndex) {
                    cumulativeNumberInstances += bags.perClassPerBag(i, j);
                }
            }
            returnValue += this.lnFunc(cumulativeNumberInstances);
            returnValue -= this.lnFunc(bags.perBag(i));
        }
        return - (returnValue / Math.log(2));
    }
    
    /**
     * Computes entropy after splitting without considering the
     * class values.
     */
    public final double splitEnt(Distribution bags) {
        double returnValue = 0.0D;
        
        for (int i = 0; i < bags.numBags(); i++) {
            returnValue += this.lnFunc(bags.perBag(i));
        }
        return (this.lnFunc(bags.total()) - returnValue) / Math.log(2);
    }
}
