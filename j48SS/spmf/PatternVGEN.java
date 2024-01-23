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
package weka.classifiers.trees.j48SS.spmf;

public class PatternVGEN implements Comparable<PatternVGEN> {
    PrefixVGEN prefix;
    public Bitmap bitmap;
    public double infoGain = 0.0D;
    public double[] infoGainsOneVsAll;
    
    public PatternVGEN(PrefixVGEN prefix, Bitmap bitmap) {
        this.prefix = prefix;
        this.bitmap = bitmap;
    }
    
    public int compareTo(PatternVGEN o) {
        if (o == this) {
            return 0;
        } else {
            int compare = o.bitmap.sidsum - this.bitmap.sidsum;
            return compare != 0 ? compare : this.hashCode() - o.hashCode();
        }
    }
    
    public int getAbsoluteSupport() {
        return this.bitmap.getSupport();
    }
}
