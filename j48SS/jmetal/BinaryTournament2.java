package weka.classifiers.trees.j48SS.jmetal;

import java.util.HashMap;
import java.util.Map;

public class BinaryTournament2 {
    private         int[]               a_;
    private         int                 index_ = 0;
    protected final Map<String, Object> parameters_;
    
    public BinaryTournament2(HashMap<String, Object> parameters) { this.parameters_ = parameters; }
    
    public int[] intPermutation(int length) {
        int[] aux    = new int[length];
        int[] result = new int[length];
        
        for (int i = 0; i < length; ++ i) {
            result[i] = i;
            aux[i]    = PseudoRandom.randInt(0, length - 1);
        }
        
        for (int i = 0; i < length; ++ i) {
            for (int j = i + 1; j < length; ++ j) {
                if (aux[i] > aux[j]) {
                    int tmp = aux[i];
                    aux[i] = aux[j];
                    aux[j] =    tmp;
                                tmp = result[i];
                    result[i] = result[j];
                    result[j] = tmp;
                }
            }
        }
        return result;
    }
    
    public Object execute(Object object) {
        SolutionSet population = (SolutionSet)object;
        if (this.index_ == 0) {
            this.a_ = intPermutation(population.size());
        }
        
        Solution solution1 = population.get(this.a_[this.index_]);
        Solution solution2 = population.get(this.a_[this.index_ + 1]);
        this.index_ = (this.index_ + 2) % population.size();
        int flag = Ranking.dominanceCompare(solution1, solution2);
        if (flag == - 1) {
            return solution1;
        }
        else if (flag == 1) {
            return solution2;
        }
        else if (solution1.getCrowdingDistance() > solution2.getCrowdingDistance()) {
            return solution1;
        }
        else if (solution2.getCrowdingDistance() > solution1.getCrowdingDistance()) {
            return solution2;
        }
        else {
            return PseudoRandom.randDouble() < 0.5D ? solution1 : solution2;
        }
    }
}
