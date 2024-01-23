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
import java.util.Iterator;
import java.util.List;

/**
 * This class represents an itemset from a sequence where the itemset is a list of strings ordered by lexical order
 * and does not contain an item twice, and the support of the itemset is not stored. Based on code from SPMF by
 * <a href="http://www.philippe-fournier-viger.com/spmf">Philippe Fournier-Viger</a>.
 *
 * @author Philippe Fournier-Viger
 * @author Martin Siddons (J48SS modified form)
 */
public class Itemset {
    /** The list of items in this itemset, lexically ordered. An item can only appear once in an itemset. */
    private final List<Integer> items = new ArrayList<>();
    
    /**
     * Constructor to create an itemset with an item
     * @param item the item
     */
    public Itemset(Integer item) {
        this.addItem(item);
    }
    
    /**
     * Constructor to create an empty itemset.
     */
    public Itemset() { }
    
    /**
     * Add an item to this itemset
     * @param value the item
     */
    public void addItem(Integer value) { this.items.add(value); }
    
    /**
     * Get the list of items
     * @return list of items
     */
    public List<Integer> getItems() { return this.items; }
    
    /**
     * Get an item at a given position in this itemset
     * @param index the position
     * @return the item
     */
    public Integer get(int index) { return this.items.get(index); }
    
    /**
     * Get this itemset as a string
     * @return this itemset as a string
     */
    public String toString() {
        StringBuilder r = new StringBuilder();
    
        for (Integer item : this.items) {
            r.append(item.toString());
            r.append(' ');
        }
        
        return r.toString();
    }
    
    /**
     * Get the size of this itemset (the number of items)
     * @return the size
     */
    public int size() { return this.items.size(); }
    
    /**
     * This method makes a copy of an itemset
     * @return the copy.
     */
    public Itemset cloneItemSet() {
        Itemset itemset = new Itemset();
        itemset.getItems().addAll(this.items);
        return itemset;
    }
    
    /**
     * This methods checks if another itemset is contained in this one.
     * @param itemset2 the other itemset
     * @return true if it is contained
     */
    public boolean containsAll(Itemset itemset2){
        // we will use this variable to remember where we are in this itemset
        int i = 0;
    
        // for each item in itemset2, we will try to find it in this itemset
        for(Integer item : itemset2.getItems()){
            boolean found = false; // flag to remember if we have find the item
        
            // we search in this itemset starting from the current position i
            while(! found && i < size()){
                if(get(i).equals(item)){
                    found = true;
                }
                // if the current item in this itemset is larger than the current item from itemset2, we return false
                // because the itemsets are assumed to be lexically ordered.
                else if(get(i) > item){
                    return false;
                }
                i++;
            }
            
            // if the item was not found in the previous loop, return false
            if(!found){
                return false;
            }
        }
        return true; // all items were found
    }
}
