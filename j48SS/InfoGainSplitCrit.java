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

import weka.core.RevisionUtils;
import weka.core.Utils;

/**
 * Class for computing the information gain for a given distribution.
 *
 * @author Eibe Frank (eibe@cs.waikato.ac.nz)
 * @version $Revision: 8034 $
 */
public final class InfoGainSplitCrit extends EntropyBasedSplitCrit {
    /** for serialization */
    private static final long serialVersionUID = 4892105020180728499L;
    
    public InfoGainSplitCrit() {}
    
    /**
     * This method is a straightforward implementation of the information gain criterion for the given distribution.
     */
    public final double splitCritValue(Distribution bags) {
        double numerator = this.oldEnt(bags) - this.newEnt(bags);
        // Splits with no gain are useless.
        // We take the reciprocal value because we want to minimize the splitting criterion's value
        return Utils.eq(numerator, 0.0D) ? Double.MAX_VALUE : bags.total() / numerator;
    }
    
    /**
     * This method computes the information gain in the same way
     * C4.5 does.
     *
     * @param bags the distribution
     * @param totalNoInst weight of ALL instances (including the
     * ones with missing values).
     */
    public final double splitCritValue(Distribution bags, double totalNoInst) {
        double noUnknown = totalNoInst - bags.total();
        double unknownRate = noUnknown / totalNoInst;
        double numerator = this.oldEnt(bags) - this.newEnt(bags);
        numerator = (1.0D - unknownRate) * numerator;
        
        // Splits with no gain are useless.
        return Utils.eq(numerator, 0.0D) ? 0.0D : numerator / bags.total();
    }
    
    /**
     * This method computes the information gain in the same way
     * C4.5 does.
     *
     * @param bags the distribution
     * @param totalNoInst weight of ALL instances
     * @param oldEnt entropy with respect to "no-split"-model.
     */
    public final double splitCritValue(Distribution bags, double totalNoInst, double oldEnt) {
        double noUnknown = totalNoInst - bags.total();
        double unknownRate = noUnknown / totalNoInst;
        double numerator = oldEnt - this.newEnt(bags);
        numerator = (1.0D - unknownRate) * numerator;
        return Utils.eq(numerator, 0.0D) ? 0.0D : numerator / bags.total();
    }
    
    /**
     * Returns the revision string.
     *
     * @return		the revision
     */
    public String getRevision() { return RevisionUtils.extract("$Revision: 10169 $"); }
    
    public final double splitCritValueOneVsAll(Distribution bags, double totalNoInst, int oneVsAllClassIndex) {
        double noUnknown = totalNoInst - bags.total();
        double unknownRate = noUnknown / totalNoInst;
        double numerator = this.oldEnt(bags, oneVsAllClassIndex) - this.newEnt(bags, oneVsAllClassIndex);
        numerator = (1.0D - unknownRate) * numerator;
        return Utils.eq(numerator, 0.0D) ? 0.0D : numerator / bags.total();
    }
}
