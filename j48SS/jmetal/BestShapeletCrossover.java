package weka.classifiers.trees.j48SS.jmetal;

import java.util.*;

/**
 * This class allows to apply a BinaryPruning crossover operator using two parent solutions.
 */
public class BestShapeletCrossover { // move to Solution?
    /** The probability of combining two individuals of the population */
    private final Double crossoverProbability_;
    
    /** Valid solution types to apply this operator */
    private static final List<Class<BestShapeletSolutionType>> VALID_TYPES =
            Collections.singletonList(BestShapeletSolutionType.class);
    
    /**
     * Constructor. Create a new SBX crossover operator with a default index given by
     * <code>DEFAULT_INDEX_CROSSOVER</code>.
     */
    public BestShapeletCrossover(Double crossoverProbability) { this.crossoverProbability_ = crossoverProbability; }
    
    /** Perform the crossover operation as defined in the J48SS paper.
     *
     * @param probability Crossover probability
     * @param parent1 The first parent
     * @param parent2 The second parent
     * @return An array containing the two offsprings
     */
    public Solution[] doCrossover(double probability, Solution parent1, Solution parent2) {
        Solution[] childrenSolution = new Solution[]{new Solution(parent1), new Solution(parent2)};
        BestShapelet child1 = childrenSolution[0].getDecisionVariables()[0];
        BestShapelet child2 = childrenSolution[1].getDecisionVariables()[0];
        
        if (PseudoRandom.randDouble() < probability) {
            int child1Len = child1.getDecimalLength();
            List<Double> child1ElementsDecimal = child1.getDecimalElements();
            List<List<Integer>> child1ElementsBinary = child1.getBinaryElements();
            int child2Len = child2.getDecimalLength();
            List<Double> child2ElementsDecimal = child2.getDecimalElements();
            List<List<Integer>> child2ElementsBinary = child2.getBinaryElements();
            
            // get a random index for the start of the tail
            int child1SwapIndex = (int)Math.round(PseudoRandom.randDouble() * (double)(child1Len - 1));
            int child2SwapIndex = (int)Math.round(PseudoRandom.randDouble() * (double)(child2Len - 1));
            
            int toBeSwappedFromChild1 = child1Len - child1SwapIndex;
            int toBeSwappedFromChild2 = child2Len - child2SwapIndex;
            int newChild1LengthDecimal = child1Len - toBeSwappedFromChild1 + toBeSwappedFromChild2;
            int newChild2LengthDecimal = child2Len - toBeSwappedFromChild2 + toBeSwappedFromChild1;
            
            child1.setDecimalLength(newChild1LengthDecimal);
            child2.setDecimalLength(newChild2LengthDecimal);
            child1.setBinaryLength(child1.intToBinaryList(newChild1LengthDecimal));
            child2.setBinaryLength(child2.intToBinaryList(newChild2LengthDecimal));
            
            // MS: gonna refactor this mess when I have a moment
            double tempDecimal;
            List<Integer> tempBinary;
            if (toBeSwappedFromChild1 >= toBeSwappedFromChild2) {
                int indexToGetFromLonger = child1Len - (toBeSwappedFromChild1 - toBeSwappedFromChild2);
                
                for(int i = 0; i < toBeSwappedFromChild1; i++) {
                    if (i + child2SwapIndex < child2ElementsDecimal.size()) {
                        tempDecimal = child2ElementsDecimal.get(i + child2SwapIndex);
                        tempBinary = child2ElementsBinary.get(i + child2SwapIndex);
                        child2ElementsDecimal.set(i + child2SwapIndex, child1ElementsDecimal.get(i + child1SwapIndex));
                        child2ElementsBinary.set(i + child2SwapIndex, child1ElementsBinary.get(i + child1SwapIndex));
                        child1ElementsDecimal.set(i + child1SwapIndex, tempDecimal);
                        child1ElementsBinary.set(i + child1SwapIndex, tempBinary);
                    } else {
                        child2ElementsDecimal.add(child1ElementsDecimal.get(indexToGetFromLonger));
                        child2ElementsBinary.add(child1ElementsBinary.get(indexToGetFromLonger));
                        child1ElementsDecimal.remove(indexToGetFromLonger);
                        child1ElementsBinary.remove(indexToGetFromLonger);
                    }
                }
            } else {
                int indexToGetFromLonger = child2Len - (toBeSwappedFromChild2 - toBeSwappedFromChild1);
                
                for(int i = 0; i < toBeSwappedFromChild2; i++) {
                    if (i + child1SwapIndex < child1ElementsDecimal.size()) {
                        tempDecimal = child1ElementsDecimal.get(i + child1SwapIndex);
                        tempBinary = child1ElementsBinary.get(i + child1SwapIndex);
                        child1ElementsDecimal.set(i + child1SwapIndex, child2ElementsDecimal.get(i + child2SwapIndex));
                        child1ElementsBinary.set(i + child1SwapIndex, child2ElementsBinary.get(i + child2SwapIndex));
                        child2ElementsDecimal.set(i + child2SwapIndex, tempDecimal);
                        child2ElementsBinary.set(i + child2SwapIndex, tempBinary);
                    } else {
                        child1ElementsDecimal.add(child2ElementsDecimal.get(indexToGetFromLonger));
                        child1ElementsBinary.add(child2ElementsBinary.get(indexToGetFromLonger));
                        child2ElementsDecimal.remove(indexToGetFromLonger);
                        child2ElementsBinary.remove(indexToGetFromLonger);
                    }
                }
            }
        }
        return childrenSolution;
    }
    
    public Object execute(Object object) throws Exception {
        Solution[] parents = (Solution[])object;
        String name;
        if (parents.length != 2) {
            name = String.class.getName();
            throw new Exception("Exception in " + name + ".execute()");
        } else if (VALID_TYPES.contains(parents[0].getType().getClass()) &&
                VALID_TYPES.contains(parents[1].getType().getClass())) {
            return this.doCrossover(this.crossoverProbability_, parents[0], parents[1]);
        } else {
            name = String.class.getName();
            throw new Exception("Exception in " + name + ".execute()");
        }
    }
}
