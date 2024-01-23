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

import java.util.BitSet;
import java.util.Collections;
import java.util.List;

public class Bitmap {
    protected static long INTERSECTION_COUNT = 0L;
    protected BitSet bitmap;
    protected int lastSID = -1;
    protected int firstItemsetID = -1;
    private int support = 0;
    protected int sidsum = 0;
    private int supportWithoutGapTotal = 0;
    
    Bitmap(int lastBitIndex) { this.bitmap = new BitSet(lastBitIndex + 1); }
    
    private Bitmap(BitSet bitmap) { this.bitmap = bitmap; }
    
    public void registerBit(int sid, int tid, List<Integer> sequencesSize) {
        int pos = sequencesSize.get(sid) + tid;
        this.bitmap.set(pos, true);
        if (sid != this.lastSID) {
            this.support++;
            this.sidsum += sid;
        }
        
        if (this.firstItemsetID == -1 || tid < this.firstItemsetID) {
            this.firstItemsetID = tid;
        }
        
        this.lastSID = sid;
    }
    
    private int bitToSID(int bit, List<Integer> sequencesSize) {
        int result = Collections.binarySearch(sequencesSize, bit);
        return result >= 0 ? result : - result - 2;
    }
    
    public int getSupport() { return this.support; }
    
    Bitmap createNewBitmapSStep(Bitmap bitmapItem, List<Integer> sequencesSize, int lastBitIndex, int maxGap) {
        Bitmap newBitmap = new Bitmap(new BitSet(lastBitIndex));
        int previousSid;
        int bitK;
        int sid;
        if (maxGap == Integer.MAX_VALUE) {
            for(previousSid = this.bitmap.nextSetBit(0); previousSid >= 0; previousSid = this.bitmap.nextSetBit(sid + 1)) {
                bitK = this.bitToSID(previousSid, sequencesSize);
                sid = this.lastBitOfSID(bitK, sequencesSize, lastBitIndex);
                boolean match = false;
                
                for(int bit = bitmapItem.bitmap.nextSetBit(previousSid + 1); bit >= 0 && bit <= sid; bit = bitmapItem.bitmap.nextSetBit(bit + 1)) {
                    newBitmap.bitmap.set(bit);
                    match = true;
                    int tid = bit - sequencesSize.get(bitK);
                    if (this.firstItemsetID == -1 || tid < this.firstItemsetID) {
                        this.firstItemsetID = tid;
                    }
                }
                
                if (match && bitK != newBitmap.lastSID) {
                    ++newBitmap.support;
                    ++newBitmap.supportWithoutGapTotal;
                    newBitmap.sidsum += bitK;
                    newBitmap.lastSID = bitK;
                }
            }
        } else {
            previousSid = -1;
            
            for(bitK = this.bitmap.nextSetBit(0); bitK >= 0; bitK = this.bitmap.nextSetBit(bitK + 1)) {
                sid = this.bitToSID(bitK, sequencesSize);
                int lastBitOfSID = this.lastBitOfSID(sid, sequencesSize, lastBitIndex);
                boolean match = false;
                boolean matchWithoutGap = false;
                
                for(int bit = bitmapItem.bitmap.nextSetBit(bitK + 1); bit >= 0 && bit <= lastBitOfSID; bit = bitmapItem.bitmap.nextSetBit(bit + 1)) {
                    matchWithoutGap = true;
                    if (bit - bitK > maxGap) {
                        break;
                    }
                    
                    newBitmap.bitmap.set(bit);
                    match = true;
                    int tid = bit - sequencesSize.get(sid);
                    if (this.firstItemsetID == -1 || tid < this.firstItemsetID) {
                        this.firstItemsetID = tid;
                    }
                }
                
                if (matchWithoutGap && previousSid != sid) {
                    ++newBitmap.supportWithoutGapTotal;
                    previousSid = sid;
                }
                
                if (match) {
                    if (sid != newBitmap.lastSID) {
                        ++newBitmap.support;
                        newBitmap.sidsum += sid;
                    }
                    newBitmap.lastSID = sid;
                }
            }
        }
        return newBitmap;
    }
    
    public int getSupportWithoutGapTotal() { return this.supportWithoutGapTotal; }
    
    private int lastBitOfSID(int sid, List<Integer> sequencesSize, int lastBitIndex) {
        return sid + 1 >= sequencesSize.size() ? lastBitIndex : sequencesSize.get(sid + 1) - 1;
    }
    
    Bitmap createNewBitmapIStep(Bitmap bitmapItem, List<Integer> sequencesSize, int lastBitIndex) {
        BitSet newBitset = new BitSet(lastBitIndex);
        Bitmap newBitmap = new Bitmap(newBitset);
        
        for(int bit = this.bitmap.nextSetBit(0); bit >= 0; bit = this.bitmap.nextSetBit(bit + 1)) {
            if (bitmapItem.bitmap.get(bit)) {
                newBitmap.bitmap.set(bit);
                int sid = this.bitToSID(bit, sequencesSize);
                if (sid != newBitmap.lastSID) {
                    newBitmap.sidsum += sid;
                    ++newBitmap.support;
                }
                
                newBitmap.lastSID = sid;
                int tid = bit - sequencesSize.get(sid);
                if (this.firstItemsetID == -1 || tid < this.firstItemsetID) {
                    this.firstItemsetID = tid;
                }
            }
        }
        return newBitmap;
    }
    
    public void setSupport(int support) { this.support = support; }
    
    public String getSIDs(List<Integer> sequencesSize) {
        StringBuilder builder = new StringBuilder();
        int lastSidSeen = -1;
        
        for(int bitK = this.bitmap.nextSetBit(0); bitK >= 0; bitK = this.bitmap.nextSetBit(bitK + 1)) {
            int sid = this.bitToSID(bitK, sequencesSize);
            if (sid != lastSidSeen) {
                if (lastSidSeen != -1) {
                    builder.append(" ");
                }
                
                builder.append(sid);
                lastSidSeen = sid;
            }
        }
        
        return builder.toString();
    }
}
