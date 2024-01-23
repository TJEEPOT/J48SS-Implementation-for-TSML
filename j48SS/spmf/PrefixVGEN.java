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

import java.util.ArrayList;
import java.util.List;

/**
 * Implementation of a prefix (a sequential pattern) as a list of itemsets as used by the VGEN algorithm
 * as used by the VGEN algorithm. Modified from code in the SPMF Data Mining Software by
 * <a href="http://www.philippe-fournier-viger.com/spmf">Philippe Fournier-Viger</a>.
 */
public class PrefixVGEN {
    // the two following variables are used for optimizations in VGEN to avoid some containment checkings
    Integer sumOfEvenItems = null; // sum of even items in this prefix
    Integer sumOfOddItems  = null;  // sumof odd items in this prefix f
    final List<Itemset> itemsets = new ArrayList<>();
    
    /**
     * Default constructor
     */
    public PrefixVGEN() {}
    
    /**
     * Make a copy of that sequence
     *
     * @return a copy of that sequence
     */
    public PrefixVGEN cloneSequence() {
        // create a new empty sequence
        PrefixVGEN sequence = new PrefixVGEN();
        // for each itemset
        for (Itemset itemset : itemsets) {
            // copy the itemset
            sequence.addItemset(itemset.cloneItemSet());
        }
        return sequence; // return the sequence
    }
    
    public Itemset get(int index)           { return this.itemsets.get(index); }
    
    public List<Itemset> getItemsets()      { return this.itemsets; }
    
    public void addItemset(Itemset itemset) { this.itemsets.add(itemset); }
    
    public int size()                       { return itemsets.size(); }
}
