package weka.classifiers.trees.j48SS.jmetal;


import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.Map.Entry;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.zip.GZIPOutputStream;


public class BestShapeletProblem {
    public static double SMALL = 1.0E-6D;
    
    protected int numberOfObjectives_ = 2;
    public ConcurrentMap<String, Double> mapOfSolutions =  new ConcurrentHashMap<>();
    public double minSplit;
    public int maxInstanceLength = -1;
    public double maxDoubleValue = -Double.MAX_VALUE;
    public Map<Integer, double[]> mapOfInstances = new HashMap<>();
    public List<String> listOfClasses = new ArrayList<>();
    public Map<String, Integer> distributionOfClasses = new HashMap<>();
    public Map<String, List<Integer>> classMapinstances = new HashMap<>();
    
    public double initialEntropy;
    protected BestShapeletSolutionType solutionType_;
    
    public BestShapeletProblem(List<String> instances, double minSplit, boolean isVerbose) {
        this.minSplit = minSplit;
        for(int i = 0; i < instances.size(); ++i) {
            String line = instances.get(i);
            String[] splitString = line.split(",");
            this.readInstanceString(splitString, i);
        }
        
        if (isVerbose){
            System.out.println("Instances read: " + this.listOfClasses.size());
        }
        this.initialEntropy = this.calculateEntropy(this.distributionOfClasses, this.listOfClasses.size());
        this.solutionType_ = new BestShapeletSolutionType(this, this.mapOfInstances, this.maxInstanceLength,
                this.maxDoubleValue);
    }
    
    public int getNumberOfObjectives() { return this.numberOfObjectives_; }

    public BestShapeletSolutionType getSolutionType() { return this.solutionType_; }
    
    public double calculateEntropy(Map<String, Integer> distribution, int numInst) {
        double entropy = 0.0D;
    
        for (String curClass : distribution.keySet()) {
            double fraction = (double)distribution.get(curClass) / (double)numInst;
            double logValue = Math.log10(fraction) / Math.log10(2.0D);
            if (fraction > 0.0D) {
                entropy += fraction * logValue;
            }
        }
        return -entropy;
    }
    
    public void readInstanceString(String[] instance, int index) {
        this.listOfClasses.add(instance[0]);
        if (this.distributionOfClasses.containsKey(instance[0])) {
            int oldCount = this.distributionOfClasses.get(instance[0]);
            this.distributionOfClasses.put(instance[0], oldCount + 1);
        } else {
            this.distributionOfClasses.put(instance[0], 1);
        }
        
        if (this.classMapinstances.containsKey(instance[0])) {
            List<Integer> oldList = this.classMapinstances.get(instance[0]);
            oldList.add(index);
            this.classMapinstances.put(instance[0], oldList);
        } else {
            List<Integer> oldList = new ArrayList<>();
            oldList.add(index);
            this.classMapinstances.put(instance[0], oldList);
        }
        
        double[] double_values = new double[instance.length - 1];
        
        for(int i = 1; i < instance.length; ++i) {
            double_values[i - 1] = Double.parseDouble(instance[i]);
            if (double_values[i - 1] > this.maxDoubleValue) {
                this.maxDoubleValue = double_values[i - 1];
            }
        }
        
        if (double_values.length > this.maxInstanceLength) {
            this.maxInstanceLength = double_values.length;
        }
        
        this.mapOfInstances.put(index, double_values);
    }
    
    
    /** Evaluate a problem and assign a solution. Should be thread-safe
     *
     * @param solution A partially complete solution object to be modified via problem evaluation
     */
    public void evaluate(Solution solution){
        BestShapelet variable = solution.getDecisionVariables()[0]; // The variable object inside solution
        String shapeletGeneDescription = variable.getShapeletGeneDescription();
        Double knownResults = this.mapOfSolutions.get(shapeletGeneDescription);
        
        double splitPoint;
        double shapeletEntropy = 0.0D;
        if (knownResults != null) {
            try {
                shapeletEntropy = knownResults;
            } catch (Exception e) {
                e.printStackTrace();
                System.out.println("Concurrency Exception!");
                System.out.println("Caused by access to the shapelet: " + shapeletGeneDescription);
                System.exit(-1);
            }
        }
        else {
            List<Double>             shapeletValues      = variable.getDecimalElements();
            HashMap<Integer, Double> instanceMapDistance = new HashMap<>();
            try {
                for (Integer instIndex : this.mapOfInstances.keySet()) {
                    double[] instTimeSeries = this.mapOfInstances.get(instIndex);
                    double   distance       = this.subsequenceDistOpt(instTimeSeries, shapeletValues);
                    instanceMapDistance.put(instIndex, distance);
                }
    
                Map<Integer, Double> sortedMap                 = this.sortHashMapByValues(instanceMapDistance);
                Map<String, Integer> mapClassDistributionLeft  = new HashMap<>();
                Map<String, Integer> mapClassDistributionRight = new HashMap<>();
        
                for (String curClass : this.distributionOfClasses.keySet()) {
                    mapClassDistributionLeft.put(curClass, 0);
                    mapClassDistributionRight.put(curClass, 0);
                }
    
                // make a count of each class and store them on the right distribution
                List<Integer> sortedInstanceIndexes = new ArrayList<>();
                for (int instIndex : sortedMap.keySet()) {
                    String instanceClass = this.listOfClasses.get(instIndex);
                    int oldValue = mapClassDistributionRight.get(instanceClass);
                    mapClassDistributionRight.put(instanceClass, oldValue + 1);
                    sortedInstanceIndexes.add(instIndex);
                }
        
                splitPoint = -Double.MAX_VALUE;
                double splitEntropy      = this.initialEntropy;
                int    numInstancesLeft  = 0;
                int    numInstancesRight = this.listOfClasses.size();
                int    lastValOfI        = 0;
                int    numClasses        = this.listOfClasses.size();
    
                for (int i = 1; i < sortedInstanceIndexes.size(); ++ i) {
                    Integer instIndex        = sortedInstanceIndexes.get(i);
                    double  currentValue     = instanceMapDistance.get(instIndex);
                    int     prevInstIndex    = sortedInstanceIndexes.get(i - 1);
                    double  prevCurrentValue = instanceMapDistance.get(prevInstIndex);
                    if (prevCurrentValue + 1.0E-5D < currentValue) {
                        for (int j = lastValOfI; j < i; ++ j) {
                            int moveIndex = sortedInstanceIndexes.get(j);
                            ++ numInstancesLeft;
                            -- numInstancesRight;
                            String instanceClass = this.listOfClasses.get(moveIndex);
    
                            // add one to the count on the left distribution, error if the count is greater than numClasses
                            int oldCountValue = mapClassDistributionLeft.get(instanceClass);
                            mapClassDistributionLeft.put(instanceClass, oldCountValue + 1);
                            if (oldCountValue + 1 > numClasses) {
                                System.err.println("Error in distributions LEFT (BestShapeletProblem).");
                                System.exit(1);
                            }
    
                            // remove one from the count of the right distribution, error if the count is now less than 0.
                            oldCountValue = mapClassDistributionRight.get(instanceClass);
                            mapClassDistributionRight.put(instanceClass, oldCountValue - 1);
                            if (oldCountValue - 1 < 0) {
                                System.err.println("Error in distributions RIGHT (BestShapeletProblem).");
                                System.exit(1);
                            }
                        }
    
                        if (grOrEq(numInstancesLeft, this.minSplit) && grOrEq(numInstancesRight, this.minSplit)) {
                            double entropyLeft  = this.calculateEntropy(mapClassDistributionLeft, numInstancesLeft);
                            double entropyRight = this.calculateEntropy(mapClassDistributionRight, numInstancesRight);
                            double currentEntropy = (double)numInstancesLeft / (double)numClasses * entropyLeft +
                                    (double)numInstancesRight / (double)numClasses * entropyRight;
                            if (splitPoint == - Double.MAX_VALUE || grOrEq(Math.abs(splitEntropy),
                                    Math.abs(currentEntropy))) {
                                splitEntropy = currentEntropy;
                                splitPoint   = currentValue;
                            }
                        }
                        lastValOfI = i;
                    }
                }
    
                if (splitPoint != -Double.MAX_VALUE) {
                    shapeletEntropy = splitEntropy;
                }
                else {
                    shapeletEntropy = this.initialEntropy;
                }
    
                if (! this.mapOfSolutions.containsKey(shapeletGeneDescription)) {
                    this.mapOfSolutions.put(shapeletGeneDescription, shapeletEntropy);
                }
            }
            catch (Exception e) {
                e.printStackTrace();
                System.exit(- 1);
            }
        }
        
        double infoGain = this.initialEntropy - shapeletEntropy;
        BestShapelet var = solution.getDecisionVariables()[0];
        List<Double> listElements = var.getDecimalElements();
        StringBuilder sb = new StringBuilder(listElements.size() * 10);
    
        for (Double elem : listElements) {
            sb.append(elem).append(",");
        }
        
        String shapeletString = sb.toString();
        String compessedShapeletString = "";
        
        try {
            compessedShapeletString = compress(shapeletString);
        } catch (Exception e) {
            e.printStackTrace();
        }
        
        splitPoint = (double)compessedShapeletString.length() / (double)shapeletString.length();
        solution.setObjective(0, infoGain * -1.0D);
        solution.setObjective(1, splitPoint);
    }
    
    public static String compress(String str) throws Exception {
        if (str != null && str.length() != 0) {
            ByteArrayOutputStream obj = new ByteArrayOutputStream();
            GZIPOutputStream gzip = new GZIPOutputStream(obj);
            gzip.write(str.getBytes(StandardCharsets.UTF_8));
            gzip.close();
            return obj.toString("UTF-8");
        } else {
            return str;
        }
    }
    
    public <K, V extends Comparable<? super V>> Map<K, V> sortHashMapByValues(Map<K, V> map) {
        List<Entry<K, V>> list = new ArrayList<>(map.entrySet());
        list.sort(Entry.comparingByValue());
        Map<K, V> result = new LinkedHashMap<>();
    
        for (Entry<K, V> kvEntry : list) {
            result.put(kvEntry.getKey(), kvEntry.getValue());
        }
        
        return result;
    }
    
    public double DTW(double[] timeSeries, int baseIndex, double[] shapelet, double bestValue) {
        double[][] distanceMatrix = new double[shapelet.length][shapelet.length];
        
        for(int i = 0; i < shapelet.length; ++i) {
            distanceMatrix[0][i] = Double.MAX_VALUE;
            distanceMatrix[i][0] = Double.MAX_VALUE;
            distanceMatrix[0][0] = 0.0D;
        }
        
        for(int i = 1; i < shapelet.length; ++i) {
            for(int j = 1; j < shapelet.length; ++j) {
                double cellCost = Math.pow(timeSeries[baseIndex + i] - shapelet[j], 2.0D);
                double minValue;
                double a = distanceMatrix[i - 1][j];
                double b = distanceMatrix[i][j - 1];
                double c = distanceMatrix[i - 1][j - 1];
                if (a < b && a < c) {
                    minValue = a;
                } else if (b < c && b < a) {
                    minValue = b;
                } else {
                    minValue = c;
                }
                
                distanceMatrix[i][j] = cellCost + minValue;
                if (distanceMatrix[i][j] > bestValue) {
                    return distanceMatrix[i][j];
                }
            }
        }
        return Math.sqrt(distanceMatrix[shapelet.length - 1][shapelet.length - 1]);
    }
    
    private double subsequenceDistOpt(double[] timeSeries, List<Double> shapelet) {
        double minimumDistance = Double.MAX_VALUE;
        int referenceTSLength = Math.max(timeSeries.length, shapelet.size());
        int shapeletSize = shapelet.size();
        
        for(int i = 0; i < referenceTSLength - shapeletSize + 1; ++i) {
            boolean stop = false;
            double sumDistance = 0.0D;
            
            for(int j = 0; j < shapeletSize; ++j) {
                double tsValue;
                if (i + j < timeSeries.length) {
                    tsValue = timeSeries[i + j];
                } else {
                    tsValue = 0.0D;
                }
                
                sumDistance += Math.pow(shapelet.get(j) - tsValue, 2.0D);
                if (sumDistance >= minimumDistance) {
                    stop = true;
                    break;
                }
            }
            
            if (!stop) {
                minimumDistance = sumDistance;
            }
        }
        
        double sqrtVal = Math.sqrt(minimumDistance);
        return round(sqrtVal, 5);
    }
    
    public static double round(double value, int places) {
        double newval = value * Math.pow(10.0D, places);
        newval = ((int)newval);
        newval /= Math.pow(10.0D, places);
        return newval;
    }
    
    public Map<Integer, double[]> getInstances() { return this.mapOfInstances; }
    
    public static boolean grOrEq(double a, double b) { return b - a < SMALL || a >= b; }
}
