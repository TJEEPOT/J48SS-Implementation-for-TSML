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

import java.util.*;
import java.util.Map.Entry;

/**
 * *
 * This is a modification of the SPMF implementation of the VGEN algorithm by
 * <a href="http://www.philippe-fournier-viger.com/spmf">Philippe Fournier-Viger</a>.
 * <br><br>
 *
 * @author Philippe Fournier-Viger &#38; Antonio Gomariz
 * @author Martin Siddons (J48SS modified form)
 * @see Bitmap
 * @see Itemset
 * @see PrefixVGEN
 * @see PatternVGEN
 */
public class AlgoVGEN {
    /** for statistics */
    public int m_patternCount;
    
    /** The minimum support that the extracted sequential patterns must have (between 0 and 1) */
    private int m_minsup = 0;
    
    /** Structure to store the horizontal database */
    List<int[]> m_inMemoryDB;
    
    /** Vertical database */
    Map<Integer, Bitmap> m_verticalDB = new HashMap<>();
    
    /** List indicating the number of bits per sequence */
    List<Integer> m_sequencesSize = null;
    
    /** the last bit position that is used in bitmaps */
    int m_lastBitIndex = 0;
    
    /** maximum pattern length in terms of item count */
    private int m_maximumPatternLength = Integer.MAX_VALUE;
    
    /**
     * Map: key: item   value:  another item that followed the first item + support
     * (could be replaced with a triangular matrix...)
     */
    Map<Integer, Map<Integer, Integer>> m_coocMapAfter  = null;
    Map<Integer, Map<Integer, Integer>> m_coocMapEquals = null;
    
    /** Map indicating for each item, the smallest tid containing this item in a sequence. */
    boolean m_useCMAPPruning = true;
    
    /** The maximum gap between two itemsets in a pattern (1 = No gap) */
    private int m_maxGap = Integer.MAX_VALUE;
    
    //================== VARIABLES THAT ARE SPECIFIC TO VGEN ===================
    /**
     * GENERATOR PATTERNS -  The list contains patterns of size k at position k in the list.
     * A map has the sum of sids as key and lists of patterns as value.
     */
    List<Map<Integer, List<PatternVGEN>>> m_generatorPatterns = null;
    
    /** if enabled, the result will be verified to see if some patterns found are not generators. */
    boolean DEBUG_MODE = false;
    
    /** the number of transaction in the database (to calculate the support of the empty set) */
    int m_transactionCount = 0;
    //=============== END OF VARIABLES THAT ARE SPECIFIC TO VGEN ===============
    
    //================== VARIABLES THAT ARE SPECIFIC TO J48SS ==================
    double[]                m_bestIG;
    HashMap<String, Double> m_bestIGforLength                  = new HashMap<>();
    HashMap<String, String> m_bestPatternforLength             = new HashMap<>();
    HashMap<String, Bitmap> m_bestPatternPrefixBitmapforLength = new HashMap<>();
    int                     m_longestPatternLength             = - 1;
    double                  m_weight;
    double                  m_lowestSupport                    = 0.0D;
    String                  m_lowestSupPattern                 = "";
    double[]                m_initialEntropy;
    double                  m_initialEntropyMulticlass         = 1.0D;
    List<String>            m_instClasses;
    HashMap<String, Double> m_classFrequencies                 = new HashMap<>();
    String[]                m_classes;
    boolean                 m_isVerbose;
    //============== END OF VARIABLES THAT ARE SPECIFIC TO J48SS ===============
    
    /**
     * Constructor.
     *
     * @param horizontalDB     replaces the input file from AlgoVGEN, since we already have the data in the system
     * @param instancesClasses list of classes in the DB
     * @param weight           weightings
     * @param isVerbose        If true, sends messages to System.out with the number of instances in the current round
     */
    public AlgoVGEN(
            List<int[]> horizontalDB, List<String> instancesClasses, double weight,
            boolean isVerbose) {
        this.m_inMemoryDB  = new ArrayList<>(horizontalDB);
        this.m_instClasses = new ArrayList<>(instancesClasses);
        this.m_weight      = weight;
        this.m_isVerbose   = isVerbose;
    }
    
    /**
     * Method to run the algorithm
     *
     * @param m_prevFoundIG the previously found IG
     * @param minsupRel     the minimum support as a relative value
     *
     * @return pattern, IG and inst_indexes for the given data
     */
    public Map<String, String> runAlgorithm(double[] m_prevFoundIG, double minsupRel) {
        if (this.DEBUG_MODE) {
            System.out.println(" %%%%%%%%%%  DEBUG MODE %%%%%%%%%%");
        }
        
        Bitmap.INTERSECTION_COUNT = 0L;
        // initialize the number of patterns found
        m_patternCount = 0;
        
        // RUN THE ALGORITHM
        this.VGEN(m_prevFoundIG, minsupRel);
        
        // ################################## FOR DEBUGGGING #############################
        // ########  THIS CODE CHECKS IF A PATTERN FOUND IS NOT A GENERATOR ##############
        if (DEBUG_MODE) {
            System.out.println("minsup absolute : " + m_minsup);
            
            List<PatternVGEN> listPatterns = new ArrayList<>();
            for (Map<Integer, List<PatternVGEN>> mapSizeI : m_generatorPatterns) {
                if (mapSizeI == null) {
                    continue;
                }
                for (List<PatternVGEN> listpattern : mapSizeI.values()) {
                    listPatterns.addAll(listpattern);
                }
            }
            // CHECK IF SOME PATTERNS ARE NOTE GENERATORS
            for (PatternVGEN pat1 : listPatterns) {
                // if this pattern is not the empty set and the support is same as empty set, then it is not a generator
                if (pat1.prefix.size() > 0 && pat1.getAbsoluteSupport() == m_transactionCount) {
                    System.out.println("NOT A GENERATOR !!!!!!!!!  " + pat1.prefix +
                            "    sup: " + pat1.bitmap.getSupport() + " because of empty set");
                }
                
                // otherwise we have to compare with every other pattern.
                for (PatternVGEN pat2 : listPatterns) {
                    if (pat1 == pat2) {
                        continue;
                    }
                    
                    if (pat1.getAbsoluteSupport() == pat2.getAbsoluteSupport()) {
                        if (strictlyContains(pat1.prefix, pat2.prefix)) {
                            System.out.println("NOT A GENERATOR !!!!!!!!!  " + pat1.prefix + " " + pat2.prefix +
                                    "   sup: " + pat1.bitmap.getSupport());
                            System.out.println(pat1.bitmap.sidsum + " " + pat2.bitmap.sidsum);
                        }
                    }
                }
            }
        }
        // ############################ END OF DEBUGGING CODE ################################
        
        // Find the info gain of the pattern as described in J48SS
        HashMap<Integer, Double>  bestOverallIGforLength    = new HashMap<>();
        HashMap<Integer, Integer> bestPatternIndexforLength = new HashMap<>();
        double                    bestIGfound               = - 1.0D;
        
        for (String key : m_bestIGforLength.keySet()) {
            if (m_bestIGforLength.get(key) > bestIGfound) {
                bestIGfound = m_bestIGforLength.get(key);
            }
            
            String[] splitKey         = key.split(";");
            int      classIndex       = Integer.parseInt(splitKey[1]);
            int      patternLength    = Integer.parseInt(splitKey[0]);
            double   patternOverallIG = this.getPatternInfoGain(m_bestPatternPrefixBitmapforLength.get(key));
            if (bestOverallIGforLength.get(patternLength) == null || patternOverallIG > bestOverallIGforLength
                    .get(patternLength)) {
                bestOverallIGforLength.put(patternLength, patternOverallIG);
                bestPatternIndexforLength.put(patternLength, classIndex);
            }
        }
        
        if (bestIGfound <= 0.0D) {
            bestIGfound = 1.0D;
        }
        
        Map<String, String> result = new HashMap<>();
        if (bestPatternIndexforLength.size() > 0) {
            int          bestPatternLength  = - 1;
            double       bestPatternWeight  = Double.MAX_VALUE;
            Set<Integer> hashMapKeySet      = bestPatternIndexforLength.keySet();
            Integer[]    hashMapKeySetArray = hashMapKeySet.toArray(new Integer[0]);
            Arrays.sort(hashMapKeySetArray);
            
            for (int patternLength : hashMapKeySetArray) {
                double weightedSum = m_weight * (1.0D - bestOverallIGforLength.get(patternLength) / bestIGfound) +
                        (1.0D - m_weight) * ((double)patternLength / (double)m_longestPatternLength);
                if (weightedSum < bestPatternWeight) {
                    bestPatternWeight = weightedSum;
                    bestPatternLength = patternLength;
                }
            }
            
            result.put("pattern", m_bestPatternforLength.get(bestPatternLength + ";" +
                    bestPatternIndexforLength.get(bestPatternLength)));
            result.put("IG", "" + bestOverallIGforLength.get(bestPatternLength));
            result.put("inst_indexes", m_bestPatternPrefixBitmapforLength.get(bestPatternLength + ";" +
                    bestPatternIndexforLength.get(bestPatternLength)).getSIDs(m_sequencesSize));
        }
        else {
            result.put("pattern", "NO_PATTERNS_FOUND_WITHIN_SUPPORT");
            result.put("IG", "-1.0");
            result.put("inst_indexes", "UNDEFINED");
        }
        
        return result;
    }
    
    private void VGEN(double[] input, double minsupRel) {
        // create maxPattern array
        m_generatorPatterns = new ArrayList<>(20);
        m_generatorPatterns.add(new HashMap<>());
        m_generatorPatterns.add(new HashMap<>());
        
        // the structure to store the vertical database
        // key: an item    value : bitmap
        m_verticalDB = new HashMap<>();
        
        // STEP 0: SCAN THE DATABASE TO STORE THE FIRST BIT POSITION OF EACH SEQUENCE
        // AND CALCULATE THE TOTAL NUMBER OF BITS FOR EACH BITMAP
        m_sequencesSize = new ArrayList<>();
        m_lastBitIndex  = 0; // variable to record the last bit position that we will use in bitmaps
        
        int bitIndex = 0;
        
        // instead of a file to be read, we have data passed in in-memory instead. Record the number of entries there
        // are in there and record the last bit position for the bitmaps
        for (int[] tokens : m_inMemoryDB) {
            m_sequencesSize.add(bitIndex);
            
            for (int token : tokens) {
                if (token == - 1) {
                    bitIndex++;
                }
            }
        }
        m_lastBitIndex = bitIndex - 1;
        
        // Calculate the absolute minimum support by multipling the percentage
        // with the number of sequences in this database
        m_minsup = (int)Math.ceil(minsupRel * (double)m_sequencesSize.size());
        if (m_minsup == 0) {
            m_minsup = 1;
        }
        
        m_transactionCount = 0;
        if (m_isVerbose) {
            System.out.println("Instances in the dataset: " + m_sequencesSize.size());
        }
        
        // STEP1: SCAN inMemoryDB TO CREATE THE BITMAP VERTICAL DATABASE REPRESENTATION
        int sid = 0; // to know which sequence we are scanning
        int tid = 0;  // to know which itemset we are scanning
        
        // for each line (sequence) from the input
        for (int[] thisLine : m_inMemoryDB) {
            // each line is formed of a set of integer tokens
            for (int token : thisLine) {
                if (token == - 1) {  // the end of an itemset
                    tid++;
                }
                else if (token == - 2) { // the end of a sequence
                    sid++;
                    tid = 0;
                }
                else { // an item
                    // Get the bitmap for this item. If none, create one.
                    Bitmap bitmapItem = m_verticalDB.get(token);
                    if (bitmapItem == null) {
                        bitmapItem = new Bitmap(m_lastBitIndex);
                        m_verticalDB.put(token, bitmapItem);
                    }
                    // Register the bit in the bitmap for this item
                    bitmapItem.registerBit(sid, tid, m_sequencesSize);
                }
            }
            m_transactionCount++;
        }
        
        // reduce the length of the IG array to the length of m_classFrequencies and save it as the bestIG
        this.determineClassFrequencies();
        m_lowestSupport = m_instClasses.size();
        m_classes       = new String[m_classFrequencies.size()];
        m_bestIG        = new double[m_classFrequencies.size()];
        
        System.arraycopy(input, 0, m_bestIG, 0, m_bestIG.length);
        
        int position = 0;
        for (String singleClass : m_classFrequencies.keySet()) {
            m_classes[position] = singleClass;
            position++;
        }
        
        m_initialEntropy = new double[m_classFrequencies.size()];
        
        for (int i = 0; i < m_classFrequencies.size(); i++) {
            HashMap<String, Double> oneVsAllClassFreqs = new HashMap<>();
            double                  otherClassesFreq   = 0.0D;
            double                  referenceClassFreq = 0.0D;
            String                  referenceClass     = m_classes[i];
            
            for (Entry<String, Double> classFreq : m_classFrequencies.entrySet()) {
                if (classFreq.getKey().equals(referenceClass)) {
                    referenceClassFreq = classFreq.getValue();
                }
                else {
                    otherClassesFreq += classFreq.getValue();
                }
            }
            
            // check if this is a multi-class problem
            if (referenceClassFreq > 0.0D) {
                oneVsAllClassFreqs.put(referenceClass, referenceClassFreq);
            }
            
            if (otherClassesFreq > 0.0D) {
                oneVsAllClassFreqs.put("otherClasses", otherClassesFreq);
            }
            
            m_initialEntropy[i] = this.determineEntropy(oneVsAllClassFreqs);
        }
        
        // STEP 2: REMOVE INFREQUENT ITEMS FROM THE DATABASE BECAUSE THEY WILL NOT APPEAR IN ANY FREQUENT SEQ. PATTERNS
        m_initialEntropyMulticlass = this.determineEntropy(m_classFrequencies);
        List<Integer> frequentItems = new ArrayList<>();
        
        for (Entry<Integer, Bitmap> entry : m_verticalDB.entrySet()) {
            if (! (entry.getValue().getSupport() < m_minsup)) {
                frequentItems.add(entry.getKey());
            }
        }
        
        // SET 2.1  SORT ITEMS BY DESCENDING SUPPORT
        frequentItems.sort(Comparator.comparingInt(arg0 -> AlgoVGEN.this.m_verticalDB.get(arg0).getSupport()));
        
        // STEP 3.1  CREATE CMAP
        m_coocMapEquals = new HashMap<>(frequentItems.size());
        m_coocMapAfter  = new HashMap<>(frequentItems.size());
        
        for (int[] transaction : m_inMemoryDB) {
            Set<Integer>               alreadyProcessed = new HashSet<>();
            Map<Integer, Set<Integer>> equalProcessed   = new HashMap<>();
            
            loopI:
            for (int i = 0; i < transaction.length; i++) {
                Integer      itemI    = transaction[i];
                Set<Integer> equalSet = equalProcessed.computeIfAbsent(itemI, k -> new HashSet<>());
                
                if (itemI < 0) {
                    continue;
                }
                
                Bitmap bitmapOfItem = m_verticalDB.get(itemI);
                if (bitmapOfItem == null || bitmapOfItem.getSupport() < m_minsup) {
                    continue;
                }
                
                Set<Integer> alreadyProcessedB = new HashSet<>();
                boolean      sameItemset       = true;
                for (int j = i + 1; j < transaction.length; j++) {
                    Integer itemJ = transaction[j];
                    
                    if (itemJ < 0) {
                        sameItemset = false;
                        continue;
                    }
                    
                    Bitmap itemJBitmap = m_verticalDB.get(itemJ);
                    if (itemJBitmap != null && itemJBitmap.getSupport() >= m_minsup) {
                        Map<Integer, Integer> map;
                        if (sameItemset) {
                            if (! equalSet.contains(itemJ)) {
                                map = m_coocMapEquals.computeIfAbsent(itemI, k -> new HashMap<>());
                                map.merge(itemJ, 1, Integer::sum);
                                equalSet.add(itemJ);
                            }
                        }
                        else if (! alreadyProcessedB.contains(itemJ)) {
                            if (alreadyProcessed.contains(itemI)) {
                                continue loopI;
                            }
                            
                            map = m_coocMapAfter.computeIfAbsent(itemI, k -> new HashMap<>());
                            map.merge(itemJ, 1, Integer::sum);
                            alreadyProcessedB.add(itemJ);
                        }
                    }
                }
                alreadyProcessed.add(itemI);
            }
        }
        
        // STEP3: WE PERFORM THE RECURSIVE DEPTH FIRST SEARCH
        // to find longer sequential patterns recursively
        if (DEBUG_MODE) {
            System.out.println("transaction count = " + m_transactionCount);
        }
        
        // NEW2014: SAVE ALL SINGLE FREQUENT ITEMS FIRST  BEFORE PERFORMING DEPTH FIRST SEARCH   =========
        List<PatternVGEN> prefixSingleItems = new ArrayList<>(m_verticalDB.entrySet().size());
        
        for (Entry<Integer, Bitmap> entry : m_verticalDB.entrySet()) {
            // We create a prefix with that item
            PrefixVGEN prefix = new PrefixVGEN();
            prefix.addItemset(new Itemset(entry.getKey()));
            boolean itemIsEven = entry.getKey() % 2 == 0;
            if (itemIsEven) {
                prefix.sumOfEvenItems = entry.getKey();
                prefix.sumOfOddItems  = 0;
            }
            else {
                prefix.sumOfEvenItems = 0;
                prefix.sumOfOddItems  = entry.getKey();
            }
            PatternVGEN pattern = new PatternVGEN(prefix, entry.getValue());
            prefixSingleItems.add(pattern);
            
            // NEW 2014 : IMPORTANT!!!! -- > DON'T OUTPUT PATTERN IF SUPPORT IS EQUAL TO SDB SIZE
            // BUT NOTE THAT WE WILL STILL NEED TO DO THE DEPTH FIRST SEARCH FOR THIS PATTERN IN THE NEXT FOR LOOP...
            if (m_transactionCount != entry.getValue().getSupport()) {
                // SAVE THE PATTERN TO THE RESULT
                List<PatternVGEN> listPatterns = m_generatorPatterns.get(1)
                        .computeIfAbsent(pattern.bitmap.sidsum, k -> new ArrayList<>());
                listPatterns.add(pattern);
                m_patternCount++;
            }
        }
        
        if (prefixSingleItems.size() > 0) {
            m_longestPatternLength = 1;
        }
        
        for (PatternVGEN pattern : prefixSingleItems) {
            double[] prefixIG = this.getPatternInfoGainOneVsAll(pattern.bitmap);
            pattern.infoGainsOneVsAll = prefixIG;
            String[] itemsetArray  = pattern.prefix.toString().split("-1");
            int      patternLength = 0;
            
            for (String itemset : itemsetArray) {
                patternLength += itemset.split(" ").length;
            }
            
            if (patternLength != 1) {
                System.out.println("Issue with singleton pattern length: " + patternLength);
                System.out.println("Pattern : " + pattern.prefix.toString());
                System.exit(0);
            }
            
            for (int i = 0; i < m_classes.length; i++) {
                if (prefixIG[i] > m_bestIG[i]) {
                    m_bestIG[i] = prefixIG[i];
                    m_bestIGforLength.put(patternLength + ";" + i, prefixIG[i]);
                    m_bestPatternforLength.put(patternLength + ";" + i, pattern.prefix.toString());
                    m_bestPatternPrefixBitmapforLength.put(patternLength + ";" + i, pattern.bitmap);
                }
            }
            if ((double)pattern.getAbsoluteSupport() < m_lowestSupport) {
                m_lowestSupport    = pattern.getAbsoluteSupport();
                m_lowestSupPattern = pattern.prefix.toString();
            }
        }
        
        // Sort the single frequent items
        prefixSingleItems.sort((pattern1, pattern2) -> {
            double maxIgPat1 = 0.0D;
            double maxIgPat2 = 0.0D;
            double avgIgPat1 = 0.0D;
            double avgIgPat2 = 0.0D;
            
            for (int i = 0; i < AlgoVGEN.this.m_classes.length; i++) {
                if (pattern1.infoGainsOneVsAll[i] > maxIgPat1) {
                    maxIgPat1 = pattern1.infoGainsOneVsAll[i];
                }
                avgIgPat1 += pattern1.infoGainsOneVsAll[i];
                
                if (pattern2.infoGainsOneVsAll[i] > maxIgPat2) {
                    maxIgPat2 = pattern2.infoGainsOneVsAll[i];
                }
                avgIgPat2 += pattern2.infoGainsOneVsAll[i];
            }
            return Double.compare(avgIgPat2, avgIgPat1);
        });
        
        // PERFORM THE DEPTH FIRST SEARCH
        for (PatternVGEN aPattern : prefixSingleItems) {
            // We create a prefix with that item
            int anitem = aPattern.prefix.get(0).get(0);
            if (m_maximumPatternLength > 1) {
                dfsPruning(aPattern.prefix, aPattern.bitmap, frequentItems, frequentItems, anitem, 2, anitem);
            }
        }
        
        // THE EMPTY SET IS ALWAYS A GENERATOR, SO ADD IT TO THE RESULT SET
        Bitmap bitmap = new Bitmap(0);
        bitmap.setSupport(m_transactionCount);
        PatternVGEN       pat        = new PatternVGEN(new PrefixVGEN(), bitmap);
        List<PatternVGEN> listLevel0 = new ArrayList<>();
        listLevel0.add(pat);
        m_generatorPatterns.get(0).put(0, listLevel0);
        m_patternCount++;
        // END NEW 2014 =============
    }
    
    /**
     * This is the dfsPruning method modified for J48SS from the version in the SPAM paper.
     *
     * @param prefix           the current prefix
     * @param prefixBitmap     the bitmap corresponding to the current prefix
     * @param sn               a list of items to be considered for i-steps
     * @param in               a list of items to be considered for s-steps
     * @param hasToBeGreaterThanForIStep the largest item already in the first itemset of prefix
     * @param m                size of the current prefix in terms of items
     * @param lastAppendedItem the last appended item to the prefix
     */
    void dfsPruning(
            PrefixVGEN prefix, Bitmap prefixBitmap, List<Integer> sn, List<Integer> in,
            int hasToBeGreaterThanForIStep, int m, Integer lastAppendedItem) {
        
        double prefixRelativeSupport = (double)prefixBitmap.getSupport() / (double)m_instClasses.size();
        double prefixPrevRelativeSupport = ((double)prefixBitmap.getSupport() - 1.0D) / (double)m_instClasses
                .size();
        if (! (prefixPrevRelativeSupport <= 0.0D)) {
            double[] prefixUpperBoundIG     = new double[m_classes.length];
            double[] prefixPrevUpperBoundIG = new double[m_classes.length];
            
            for (int i = 0; i < m_classes.length; i++) {
                double classFreq = m_classFrequencies.get(m_classes[i]);
                prefixUpperBoundIG[i]     = m_initialEntropy[i] -
                        this.condEntropyLBBinary(classFreq, prefixRelativeSupport);
                prefixPrevUpperBoundIG[i] = m_initialEntropy[i] -
                        this.condEntropyLBBinary(classFreq, prefixPrevRelativeSupport);
            }
            
            boolean allStop = true;
            
            for (int i = 0; i < m_classes.length; i++) {
                if (! (prefixPrevUpperBoundIG[i] <= prefixUpperBoundIG[i])) {
                    allStop = false;
                    break;
                }
                
                if (prefixUpperBoundIG[i] > m_bestIG[i]) {
                    allStop = false;
                    break;
                }
            }
            
            if (! allStop) {
                double[] prefixIG      = this.getPatternInfoGainOneVsAll(prefixBitmap);
                String[] itemsetArray  = prefix.toString().split("-1");
                int      patternLength = 0;
                
                for (String itemset : itemsetArray) {
                    patternLength += itemset.split(" ").length;
                }
                
                for (int i = 0; i < m_classes.length; i++) {
                    if (prefixIG[i] > m_bestIG[i]) {
                        m_bestIG[i] = prefixIG[i];
                    }
                    
                    if (m_bestIGforLength
                            .get(patternLength + ";" + i) == null || prefixIG[i] > m_bestIGforLength
                            .get(patternLength + ";" + i)) {
                        m_bestIGforLength.put(patternLength + ";" + i, prefixIG[i]);
                        m_bestPatternforLength.put(patternLength + ";" + i, prefix.toString());
                        m_bestPatternPrefixBitmapforLength.put(patternLength + ";" + i, prefixBitmap);
                    }
                    
                    if (patternLength > m_longestPatternLength) {
                        m_longestPatternLength = patternLength;
                    }
                }
                
                if ((double)prefixBitmap.getSupport() < m_lowestSupport) {
                    m_lowestSupport    = prefixBitmap.getSupport();
                    m_lowestSupPattern = prefix.toString();
                }
                
                //  ======  S-STEPS ======
                // Temporary variables (as described in the paper)
                List<Integer> sTemp        = new ArrayList<>();
                List<Bitmap>  sTempBitmaps = new ArrayList<>();
                
                // for CMAP pruning, we will only check against the last appended item
                Map<Integer, Integer> mapSupportItemsAfter = m_coocMapAfter.get(lastAppendedItem);
                
                loopi:
                for (Integer i : sn) {
                    
                    // CMAP PRUNING
                    // we only check with the last appended item
                    if (m_useCMAPPruning) {
                        if (mapSupportItemsAfter == null) {
                            continue loopi;
                        }
                        Integer support = mapSupportItemsAfter.get(i);
                        if (support == null || support < m_minsup) {
                            continue loopi;
                        }
                    }
                    
                    // perform the S-STEP with that item to get a new bitmap
                    Bitmap.INTERSECTION_COUNT++;
                    Bitmap newBitmap = prefixBitmap.createNewBitmapSStep(m_verticalDB.get(i), m_sequencesSize,
                            m_lastBitIndex, m_maxGap);
                    // if the support is higher than minsup
                    if (newBitmap.getSupportWithoutGapTotal() >= m_minsup) {
                        // record that item and pattern in temporary variables
                        sTemp.add(i);
                        sTempBitmaps.add(newBitmap);
                    }
                }
                
                // for each pattern recorded for the s-step
                for (int k = 0; k < sTemp.size(); k++) {
                    
                    int item = sTemp.get(k);
                    // create the new prefix
                    PrefixVGEN prefixSStep = prefix.cloneSequence();
                    prefixSStep.addItemset(new Itemset(item));
                    if (item % 2 == 0) {
                        prefixSStep.sumOfEvenItems = item + prefix.sumOfEvenItems;
                        prefixSStep.sumOfOddItems  = prefix.sumOfOddItems;
                    }
                    else {
                        prefixSStep.sumOfEvenItems = prefix.sumOfEvenItems;
                        prefixSStep.sumOfOddItems  = item + prefix.sumOfOddItems;
                    }
                    
                    // create the new bitmap
                    Bitmap newBitmap = sTempBitmaps.get(k);
                    
                    // save the pattern
                    if (newBitmap.getSupport() >= m_minsup) {
                        // NEW STRATEGY :  IMMEDIATE BACKWARD EXTENSION
                        if (m_maximumPatternLength > m) {
                            boolean hasBackWardExtension = this.savePatternMultipleItems(prefixSStep, newBitmap, m);
                            // NEW 2014: IF BACKWARD EXTENSION, THEN WE DON'T CONTINUE...
                            if (! hasBackWardExtension) {
                                dfsPruning(prefixSStep, newBitmap, sTemp, sTemp, item, m + 1, item);
                            }
                        }
                    }
                }
                
                Map<Integer, Integer> mapSupportItemsEquals = m_coocMapEquals.get(lastAppendedItem);
                // ========  I STEPS =======
                // Temporary variables
                List<Integer> iTemp        = new ArrayList<>();
                List<Bitmap>  iTempBitmaps = new ArrayList<>();
                
                // for each item in in
                loop2:
                for (Integer i : in) {
                    // the item has to be greater than the largest item already in the last itemset of prefix.
                    if (i > hasToBeGreaterThanForIStep) {
                        // CMAP PRUNING
                        if (m_useCMAPPruning) {
                            if (mapSupportItemsEquals == null) {
                                continue loop2;
                            }
                            Integer support = mapSupportItemsEquals.get(i);
                            if (support == null || support < m_minsup) {
                                continue loop2;
                            }
                        }
                        
                        // Perform an i-step with this item and the current prefix. This creates a new bitmap.
                        Bitmap.INTERSECTION_COUNT++;
                        Bitmap newBitmap = prefixBitmap.createNewBitmapIStep(m_verticalDB.get(i), m_sequencesSize,
                                m_lastBitIndex);
                        
                        // If the support is no less than minsup, record that item and pattern in temporary variables
                        if (newBitmap.getSupport() >= m_minsup) {
                            iTemp.add(i);
                            iTempBitmaps.add(newBitmap);
                        }
                    }
                }
                
                // for each pattern recorded for the i-step
                for (int k = 0; k < iTemp.size(); k++) {
                    int item = iTemp.get(k);
                    
                    // create the new prefix
                    PrefixVGEN prefixIStep = prefix.cloneSequence();
                    prefixIStep.getItemsets().get(prefixIStep.size() - 1).addItem(item);
                    
                    if (item % 2 == 0) {
                        prefixIStep.sumOfEvenItems = item + prefix.sumOfEvenItems;
                        prefixIStep.sumOfOddItems  = prefix.sumOfOddItems;
                    }
                    else {
                        prefixIStep.sumOfEvenItems = prefix.sumOfEvenItems;
                        prefixIStep.sumOfOddItems  = item + prefix.sumOfOddItems;
                    }
                    
                    // create the new bitmap
                    Bitmap newBitmap = iTempBitmaps.get(k);
                    
                    // NEW STRATEGY :  IMMEDIATE BACKWARD EXTENSION
                    if (m_maximumPatternLength > m) {
                        boolean hasBackWardExtension = savePatternMultipleItems(prefixIStep, newBitmap, m);
                        // NEW 2014: IF NO BACKWARD EXTENSION, THEN WE TRY TO EXTEND THAT PATTERN
                        if (! hasBackWardExtension) {
                            dfsPruning(prefixIStep, newBitmap, sTemp, iTemp, item, m + 1, item);
                        }
                    }
                }
            }
        }
    }
    
    /**
     * Save a pattern where size is greater than 1 to the output file.
     *
     * @param prefix the prefix
     * @param bitmap its bitmap
     * @param length size of the current prefix in terms of items
     *
     * @return true IF THE PATTERN HAS A BACKWARD EXTENSION WITH THE SAME PROJECTED DATABASE
     */
    private boolean savePatternMultipleItems(PrefixVGEN prefix, Bitmap bitmap, int length) {
        int sidsum = bitmap.sidsum;
        
        // IF THE SUPPORT OF THIS PATTERN "PREFIX" IS THE SUPPORT OF THE EMPTY SET, THEN
        // THIS PATTERN IS NOT A GENERATOR.
        if (bitmap.getSupport() == m_transactionCount) {
            return false;
        }
        
        // WE COMPARE PATTERN "PREFIX" WITH SMALLER PATTERNS FOR SUB-PATTERN CHECKING
        boolean mayBeAGenerator = true;
        // FOR PATTERNS OF SIZE 1 TO THE SIZE OF THE PATTERN MINUS 1
        for (int i = 1; i < length && i < m_generatorPatterns.size(); i++) {
            // GET ALL THE PATTERNS HAVING THE SAME SID-SUM AS THE CURRENT PATTERN
            List<PatternVGEN> level = m_generatorPatterns.get(i).get(sidsum);
            if (level == null) {
                continue;
            }
            for (PatternVGEN pPrime : level) {
                
                // CHECK THE SUM OF EVEN AND ODD ITEMS AND THE SUPPORT
                if (prefix.sumOfEvenItems >= pPrime.prefix.sumOfEvenItems &&
                        prefix.sumOfOddItems >= pPrime.prefix.sumOfOddItems &&
                        bitmap.getSupport() == pPrime.getAbsoluteSupport() &&
                        strictlyContains(prefix, pPrime.prefix)) {
                    
                    // CHECK HERE IF THERE IS A BACKWARD EXTENSION...
                    if (isThereBackwardExtension(bitmap, pPrime.bitmap)) {
                        // THERE IS A BACKWARD EXTENSION SO WE RETURN TRUE TO PRUNE EXTENSIONS
                        // OF THE PATTERN "PREFIX"
                        return true;
                    }
                    else {
                        // WE FLAG THE PATTERN "PREFIX" HAS NOT BEING A GENERATOR BUT
                        // WE CONTINUE COMPARING WITH OTHER PATTERNS TO SEE IF WE COULD PRUNE
                        mayBeAGenerator = false;
                    }
                    // END IMPORTANT
                }
            }
        }
        
        if (! mayBeAGenerator) {
            return false;
        }
        
        // WE COMPARE WITH LARGER PATTERNS FOR SUPER-PATTERN CHECKING
        for (int i = m_generatorPatterns.size() - 1; i > length; i--) {
            
            List<PatternVGEN> level = m_generatorPatterns.get(i).get(sidsum);
            if (level == null) {
                continue;
            }
            for (PatternVGEN pPrime : level) {
                
                if (prefix.sumOfEvenItems <= pPrime.prefix.sumOfEvenItems &&
                        prefix.sumOfOddItems <= pPrime.prefix.sumOfOddItems &&
                        bitmap.getSupport() == pPrime.getAbsoluteSupport() &&
                        strictlyContains(pPrime.prefix, prefix)) {
                    
                    m_patternCount--;  // DECREASE COUNT
                }
            }
        }
        
        // OTHERWISE THE PATTERN "PREFIX" MAY BE A GENERATOR SO WE KEEP IT
        while (m_generatorPatterns.size() - 1 < length) {
            m_generatorPatterns.add(new HashMap<>());
        }
        
        List<PatternVGEN> listPatterns = m_generatorPatterns.get(length)
                .computeIfAbsent(sidsum, k -> new ArrayList<>());
        
        m_patternCount++;  // INCREASE COUNT
        listPatterns.add(new PatternVGEN(prefix, bitmap));
        
        return false; // No backward extension has been found.
    }
    
    /**
     * Check if there is a backward extension by comparing the bitmap of two patterns
     * P1 and P2, such that P1 is a superset of P2
     *
     * @param bitmap1 bitmap of P1
     * @param bitmap2 bitmap of P2
     *
     * @return true if there is a backward extension
     */
    private boolean isThereBackwardExtension(Bitmap bitmap1, Bitmap bitmap2) {
        BitSet bitset1     = bitmap1.bitmap;
        BitSet bitset2     = bitmap2.bitmap;
        int    currentBit1 = bitset1.nextSetBit(0);
        int    currentBit2 = bitset2.nextSetBit(0);
        
        do {
            if (currentBit1 > currentBit2) {
                return false;
            }
            
            currentBit1 = bitset1.nextSetBit(currentBit1 + 1);
            currentBit2 = bitset2.nextSetBit(currentBit2 + 1);
        }while (currentBit1 > 0);
        
        return true;
    }
    
    /**
     * This methods checks if a seq. pattern "pattern2" is strictly contained in a seq. pattern "pattern1".
     *
     * @param pattern1 a sequential pattern
     * @param pattern2 another sequential pattern
     *
     * @return true if the pattern1 contains pattern2.
     */
    boolean strictlyContains(PrefixVGEN pattern1, PrefixVGEN pattern2) {
        // To see if pattern2 is strictly contained in pattern1, we will search for each itemset i of pattern2 in
        // pattern1 by advancing in pattern 1 one itemset at a time.
        int i = 0;
        int j = 0;
        
        while (true) {
            // if the itemset at current position in pattern1 contains the itemset at current position in pattern2
            if (pattern1.get(j).containsAll(pattern2.get(i))) {
                // go to next itemset in pattern2
                i++;
                
                // if we reached the end of pattern2, then return true
                if (i == pattern2.size()) {
                    return true;
                }
            }
            
            // go to next itemset in pattern1
            j++;
            
            // if we reached the end of pattern1, then pattern2 is not strictly included in it, return false
            if (j >= pattern1.size()) {
                return false;
            }
            
            // lastly, for optimization, we check how many itemsets are left to be matched. If there is less itemsets
            // left in pattern1 than in pattern2, then it will be impossible to get a  total match, so we return false.
            if (pattern1.size() - j < pattern2.size() - i) {
                return false;
            }
        }
    }
    
    /**
     * Set the maximum length of patterns to be found (in terms of itemset count)
     *
     * @param v the maximumPatternLength to set
     */
    public void setMaximumPatternLength(int v) { m_maximumPatternLength = v; }
    
    /**
     * This method allows to specify the maximum gap between itemsets of patterns found by the algorithm. If set to 1,
     * only patterns of contiguous itemsets will be found (no gap).
     *
     * @param v the maximum gap (an integer)
     */
    public void setMaxGap(int v) { m_maxGap = v; }
    
    
    public void determineClassFrequencies() {
        double numInstances = m_instClasses.size();
        
        for (String aClass : m_instClasses) {
            if (m_classFrequencies.containsKey(aClass)) {
                double prevVal = m_classFrequencies.get(aClass);
                prevVal += 1.0D / numInstances;
                m_classFrequencies.put(aClass, prevVal);
            }
            else {
                m_classFrequencies.put(aClass, 1.0D / numInstances);
            }
        }
    }
    
    public double determineEntropy(HashMap<String, Double> classFreqsIn) {
        double entropy = 0.0D;
        
        for (String singleClass : classFreqsIn.keySet()) {
            double frequency = classFreqsIn.get(singleClass);
            frequency *= Math.log(frequency) / Math.log(2.0D);
            entropy += frequency;
        }
        return entropy * - 1.0D;
    }
    
    public double condEntropyLBBinary(double p, double t) {
        double condEntropyLB;
        double condEntropyLB1;
        double condEntropyLB2;
        if (t <= p) {
            condEntropyLB1 = (t - 1.0D) * ((p - t) / (1.0D - t) * (Math.log((p - t) / (1.0D - t)) / Math
                    .log(2.0D)) + (1.0D - p) / (1.0D - t) * (Math.log((1.0D - p) / (1.0D - t)) / Math.log(2.0D)));
            condEntropyLB2 = - p * (Math.log(p / (1.0D - t)) / Math.log(2.0D)) + (t - 1.0D + p) * (Math
                    .log((1.0D - p - t) / (1.0D - t)) / Math.log(2.0D));
        }
        else {
            condEntropyLB1 = - p * (Math.log(p / t) / Math.log(2.0D)) - (t - p) * (Math.log(1.0D - p / t) / Math
                    .log(2.0D));
            condEntropyLB2 = - (t - 1.0D + p) * (Math.log((t - 1.0D + p) / t) / Math.log(2.0D)) - (1.0D - p) * (Math
                    .log((1.0D - p) / t) / Math.log(2.0D));
        }
        condEntropyLB = Math.min(condEntropyLB1, condEntropyLB2);
        
        if (Double.isNaN(condEntropyLB)) {
            condEntropyLB = 0.0D;
        }
        
        return condEntropyLB;
    }
    
    
    /** Find the Information Gain of the given pattern (as described in J48SS paper)
     *
     * @param prefixBitmap the pattern to get infoGain from, in Bitmap format.
     *
     * @return The information gain of pattern prefixBitmap
     */
    public double getPatternInfoGain(Bitmap prefixBitmap) {
        double    numInstances        = m_instClasses.size();
        double    nSeqsWithPattern    = prefixBitmap.getSupport();
        double    freqSeqsWithPattern = nSeqsWithPattern / numInstances;
        boolean[] seqsWithPatternBool = new boolean[(int)numInstances];
        String[]  seqsWithPattern     = prefixBitmap.getSIDs(m_sequencesSize).split(" ");
        
        for (String sid : seqsWithPattern) {
            seqsWithPatternBool[Integer.parseInt(sid)] = true;
        }
        
        HashMap<String, Double> classFreqsWithPattern    = new HashMap<>();
        HashMap<String, Double> classFreqsWithoutPattern = new HashMap<>();
        
        double prevVal;
        for (int i = 0; (double)i < numInstances; i++) {
            String instClass;
            if (seqsWithPatternBool[i]) {
                instClass = m_instClasses.get(i);
                if (classFreqsWithPattern.containsKey(instClass)) {
                    prevVal = classFreqsWithPattern.get(instClass);
                    prevVal += 1.0D / nSeqsWithPattern;
                    classFreqsWithPattern.put(instClass, prevVal);
                }
                else {
                    classFreqsWithPattern.put(instClass, 1.0D / nSeqsWithPattern);
                }
            }
            else {
                instClass = m_instClasses.get(i);
                if (classFreqsWithoutPattern.containsKey(instClass)) {
                    prevVal = classFreqsWithoutPattern.get(instClass);
                    prevVal += 1.0D / (numInstances - nSeqsWithPattern);
                    classFreqsWithoutPattern.put(instClass, prevVal);
                }
                else {
                    classFreqsWithoutPattern.put(instClass, 1.0D / (numInstances - nSeqsWithPattern));
                }
            }
        }
        
        double entropySeqsWithPattern = this.determineEntropy(classFreqsWithPattern);
        prevVal = this.determineEntropy(classFreqsWithoutPattern);
        double splitEntropy = freqSeqsWithPattern * entropySeqsWithPattern + (1.0D - freqSeqsWithPattern) * prevVal;
        if (Double.isNaN(splitEntropy)) {
            System.out.println("NaN found!!!");
            System.out.println(classFreqsWithPattern);
            System.out.println(classFreqsWithoutPattern);
            System.exit(1);
        }
        
        return m_initialEntropyMulticlass - splitEntropy;
    }
    
    public double[] getPatternInfoGainOneVsAll(Bitmap prefixBitmap) {
        double    numInstances        = m_instClasses.size();
        double    nSeqsWithPattern    = prefixBitmap.getSupport();
        double    freqSeqsWithPattern = nSeqsWithPattern / numInstances;
        boolean[] seqsWithPatternBool = new boolean[(int)numInstances];
        String[]  seqsWithPattern     = prefixBitmap.getSIDs(m_sequencesSize).split(" ");
        
        for (String sid : seqsWithPattern) {
            seqsWithPatternBool[Integer.parseInt(sid)] = true;
        }
        
        HashMap<String, Double> classFreqsWithPattern    = new HashMap<>();
        HashMap<String, Double> classFreqsWithoutPattern = new HashMap<>();
        
        for (int i = 0; i < numInstances; i++) {
            double prevVal;
            String instClass;
            if (seqsWithPatternBool[i]) {
                instClass = m_instClasses.get(i);
                if (classFreqsWithPattern.containsKey(instClass)) {
                    prevVal = classFreqsWithPattern.get(instClass);
                    prevVal += 1.0D / nSeqsWithPattern;
                    classFreqsWithPattern.put(instClass, prevVal);
                }
                else {
                    classFreqsWithPattern.put(instClass, 1.0D / nSeqsWithPattern);
                }
            }
            else {
                instClass = m_instClasses.get(i);
                if (classFreqsWithoutPattern.containsKey(instClass)) {
                    prevVal = classFreqsWithoutPattern.get(instClass);
                    prevVal += 1.0D / (numInstances - nSeqsWithPattern);
                    classFreqsWithoutPattern.put(instClass, prevVal);
                }
                else {
                    classFreqsWithoutPattern.put(instClass, 1.0D / (numInstances - nSeqsWithPattern));
                }
            }
        }
        
        double[] oneVsAllGains = new double[m_classFrequencies.size()];
        
        for (int i = 0; i < m_classes.length; i++) {
            String                  referenceClass                   = m_classes[i];
            HashMap<String, Double> oneVsAllClassFreqsWithPattern    = new HashMap<>();
            HashMap<String, Double> oneVsAllClassFreqsWithoutPattern = new HashMap<>();
            double                  otherClassesFreq                 = 0.0D;
            double                  referenceClassFreq               = 0.0D;
            
            for (Entry<String, Double> entry : classFreqsWithoutPattern.entrySet()) {
                if (entry.getKey().equals(referenceClass)) {
                    referenceClassFreq = entry.getValue();
                }
                else {
                    otherClassesFreq += entry.getValue();
                }
            }
            
            if (referenceClassFreq > 0.0D) {
                oneVsAllClassFreqsWithPattern.put(referenceClass, referenceClassFreq);
            }
            else {
                oneVsAllClassFreqsWithoutPattern.put(referenceClass, referenceClassFreq);
            }
            
            if (otherClassesFreq > 0.0D) {
                oneVsAllClassFreqsWithPattern.put("otherClasses", otherClassesFreq);
            }
            else {
                oneVsAllClassFreqsWithoutPattern.put("otherClasses", otherClassesFreq);
            }
            
            double entropySeqsWithPattern    = this.determineEntropy(oneVsAllClassFreqsWithPattern);
            double entropySeqsWithoutPattern = this.determineEntropy(oneVsAllClassFreqsWithoutPattern);
            double splitEntropy = freqSeqsWithPattern * entropySeqsWithPattern +
                    (1.0D - freqSeqsWithPattern) * entropySeqsWithoutPattern;
            
            if (Double.isNaN(splitEntropy)) {
                System.out.println("Nan found!!!");
                System.out.println(classFreqsWithPattern);
                System.out.println(classFreqsWithoutPattern);
                System.out.println(oneVsAllClassFreqsWithPattern);
                System.out.println(oneVsAllClassFreqsWithoutPattern);
                System.exit(1);
            }
            oneVsAllGains[i] = m_initialEntropy[i] - splitEntropy;
        }
        
        return oneVsAllGains;
    }
}
