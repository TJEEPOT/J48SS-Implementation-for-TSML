package weka.classifiers.trees.j48SS.jmetal;

import java.util.*;

public class BestShapeletMutation {
    private static final List<Class<BestShapeletSolutionType>> VALID_TYPES =
            Collections.singletonList(BestShapeletSolutionType.class);
    
    /** The probability of an element to undergo a random mutation */
    private final Double mutationProbability_;
    
    public BestShapeletMutation(Double probability) { this.mutationProbability_ = probability; }
    
    public void doMutation(double probability, Solution solution) {
        if (PseudoRandom.randDouble() < probability) {
            double random = PseudoRandom.randDouble(); // weirdly, we need to do this or things break...
            this.elementsMutation(probability, solution);
        }
    }
    
    private void elementsMutation(double probability, Solution solution) {
        BestShapelet solutionContent = solution.getDecisionVariables()[0];
        List<Double> elementsDouble = solutionContent.getDecimalElements();
        List<List<Integer>> elementsBinary = solutionContent.getBinaryElements();
        
        for(int i = 0; i < elementsBinary.size(); i++) {
            List<Integer> element = elementsBinary.get(i);
            if (element.size() != 64) {
                System.err.println("ERROR: wrong IEEE 754 size -> " + element.size());
            }
            
            int exponentOneCounter = 0;
            
            double convertedDoubleValue;
            for(int j = 63; j >= 0; j--) {
                convertedDoubleValue = Math.log((65 - (j + 1))) / Math.log(65.0D);
                double curProbability = probability - probability * convertedDoubleValue;
                if (PseudoRandom.randDouble() < curProbability) {
                    int oldBit = element.get(j);
                    element.set(j, (oldBit + 1) % 2);
                    if (j > 0 && j < 12 && oldBit == 0) {
                        exponentOneCounter++;
                    }
                } else if (j > 0 && j < 12 && element.get(j) == 1) {
                    exponentOneCounter++;
                }
            }
            
            if (exponentOneCounter == 11) {
                int random = (int)Math.round(PseudoRandom.randDouble() * 10.0D) + 1;
                element.set(random, 0);
            }
            convertedDoubleValue = solutionContent.binaryListToDouble(element);
            elementsDouble.set(i, convertedDoubleValue);
        }
    }
    
    public Object execute(Object object) throws Exception {
        Solution solution = (Solution)object;
        if (!VALID_TYPES.contains(solution.getType().getClass())) {
            Class<String> cls  = String.class;
            String        name = cls.getName();
            throw new Exception("Exception in " + name + ".execute()");
        } else {
            this.doMutation(this.mutationProbability_, solution);
            return solution;
        }
    }
}
