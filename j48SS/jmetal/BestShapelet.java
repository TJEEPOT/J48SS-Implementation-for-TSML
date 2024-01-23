package weka.classifiers.trees.j48SS.jmetal;

import java.math.BigInteger;
import java.util.*;

/**
 * This class implements a double array, which represents a shapelet
 */
public class BestShapelet {
    public double maxInstanceValue_;
    public int maxInstanceLength_;
    public Map<Integer, double[]> mapOfInstances_;
    List<Integer> shapeletLengthBinary;
    int shapeletLengthDecimal;
    List<List<Integer>> shapeletElementsBinary;
    List<Double> shapeletElementsDecimal;
    
    /**
     * Default constructor. Creates a new shapelet from the dataset.
     */
    public BestShapelet(Map<Integer, double[]> mapOfInstances, int maxInstanceLength, double maxInstanceValue) {
        this.maxInstanceValue_ = maxInstanceValue;
        this.maxInstanceLength_ = maxInstanceLength;
        this.mapOfInstances_ = mapOfInstances;
        int      instanceindexInt = (int)Math.round(PseudoRandom.randDouble() * (double)(mapOfInstances.keySet().size() - 1));
        double[] timeSeries       = mapOfInstances.get(instanceindexInt);
        int      beginIndex       = (int)Math.round(PseudoRandom.randDouble() * (double)(timeSeries.length - 1));
        int      endIndex         = (int)Math.round(PseudoRandom.randDouble() * (double)(timeSeries.length - 1));
        if (endIndex < beginIndex) {
            int temp = beginIndex;
            beginIndex = endIndex;
            endIndex = temp;
        }
        
        this.shapeletLengthBinary = new ArrayList<>();
        this.shapeletLengthDecimal = endIndex - beginIndex + 1;
        String binaryLengthString = Integer.toBinaryString(this.shapeletLengthDecimal);
        int maxBitsNecessary = Integer.toBinaryString(this.maxInstanceLength_).length();
        int paddingLength = maxBitsNecessary - binaryLengthString.length();
        int stringIndex = 0;
        
        for(int i = 0; i < maxBitsNecessary; ++i) {
            if (paddingLength > 0) {
                this.shapeletLengthBinary.add(0);
                --paddingLength;
            } else {
                this.shapeletLengthBinary.add(Integer.parseInt("" + binaryLengthString.charAt(stringIndex)));
                ++stringIndex;
            }
        }
        
        this.shapeletElementsBinary = new ArrayList<>();
        this.shapeletElementsDecimal = new ArrayList<>();
        
        for(int i = beginIndex; i <= endIndex; ++i) {
            double tsValue = timeSeries[i];
            this.shapeletElementsDecimal.add(tsValue);
            this.shapeletElementsBinary.add(this.doubleToBinaryList(tsValue));
        }
        
    }
    
    public Double binaryListToDouble(List<Integer> list) {
        StringBuilder sb = new StringBuilder(list.size() * 10);
    
        for (Integer num : list) {
            sb.append(num);
        }
        
        String stringList = sb.toString();
        return Double.longBitsToDouble((new BigInteger(stringList, 2)).longValue());
    }
    
    public List<Integer> doubleToBinaryList(double doub) {
        StringBuilder binaryString = new StringBuilder(Long.toBinaryString(Double.doubleToRawLongBits(doub)));
        int           i;
        if (binaryString.length() < 64) {
            int padding = 64 - binaryString.length();
            
            for(i = 0; i < padding; ++i) {
                binaryString.insert(0, "0");
            }
        }
        
        List<Integer> binaryElement = new ArrayList<>();
        
        for(i = 0; i < binaryString.length(); ++i) {
            binaryElement.add(Integer.parseInt("" + binaryString.charAt(i)));
        }
        
        return binaryElement;
    }
    
    public List<Integer> intToBinaryList(int whole) {
        List<Integer> resultList = new ArrayList<>();
        String binaryRepr = Integer.toBinaryString(whole);
        
        for(int i = 0; i < binaryRepr.length(); ++i) {
            resultList.add(Integer.parseInt(String.valueOf(binaryRepr.charAt(i))));
        }
        
        return resultList;
    }
    
    public String getShapeletGeneDescription() {
        return "L" + this.shapeletLengthDecimal + "B" + this.listAsString(this.shapeletElementsDecimal);
    }
    
    public int getDecimalLength() {
        return this.shapeletLengthDecimal;
    }
    
    public void setDecimalLength(int length) {
        this.shapeletLengthDecimal = length;
    }
    
    public void setBinaryLength(List<Integer> length) {
        this.shapeletLengthBinary = length;
    }
    
    public List<List<Integer>> getBinaryElements() {
        return this.shapeletElementsBinary;
    }
    
    public List<Double> getDecimalElements() { return this.shapeletElementsDecimal; }

    public BestShapelet(BestShapelet variable) {
        this.maxInstanceValue_ = variable.maxInstanceValue_;
        this.maxInstanceLength_ = variable.maxInstanceLength_;
        this.mapOfInstances_ = variable.mapOfInstances_;
        this.shapeletLengthDecimal = variable.shapeletLengthDecimal;
        this.shapeletLengthBinary = new ArrayList<>(variable.shapeletLengthBinary);
        this.shapeletElementsBinary = new ArrayList<>(variable.shapeletElementsBinary);
        this.shapeletElementsDecimal = new ArrayList<>(variable.shapeletElementsDecimal);
    }
    
    /**
     * This method is intended to be used in subclasses of <code>Binary</code>, for example the classes,
     * <code>BinaryReal</code> and <code>BinaryInt</code>. In these classes, the method allows us to decode the
     * value encoded in the binary string. As generic variables do not encode any value, this method does nothing
     */
    public void decode() {
    }
    
    public BestShapelet deepCopy() {
        return new BestShapelet(this);
    }
    
    public String toString() {
        return this.listAsString(this.shapeletElementsDecimal);
    }
    
    public String listAsString(List<Double> list) {
        StringBuilder sb = new StringBuilder(list.size() * 10);
        for (Double d : list) {
            sb.append(d).append(",");
        }
        
        return sb.substring(0, sb.toString().length() - 1);
    }
}
