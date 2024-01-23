package weka.classifiers.trees.j48SS.jmetal;

import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;

public class Ranking {
    private final SolutionSet[] ranking_;
    
    public Ranking(SolutionSet solutionSet) {
        int[]  dominateMe = new int[solutionSet.size()];
        List[] iDominate = new List[solutionSet.size()];
        List[] front     = new List[solutionSet.size() + 1];
        
        
        for (int i = 0; i < front.length; ++ i) {
            front[i] = new LinkedList<>();
        }
        
        for (int i = 0; i < solutionSet.size(); ++ i) {
            iDominate[i]  = new LinkedList<Integer>();
            dominateMe[i] = 0;
        }
        
        for (int i = 0; i < solutionSet.size() - 1; ++ i) {
            for (int j = i + 1; j < solutionSet.size(); ++ j) {
                int flagDominate = constraintViolationCompare(solutionSet.get(i), solutionSet.get(j));
                if (flagDominate == 0) {
                    flagDominate = dominanceCompare(solutionSet.get(i), solutionSet.get(j));
                }
                
                if (flagDominate == - 1) {
                    iDominate[i].add(j);
                    dominateMe[j]++;
                }
                else if (flagDominate == 1) {
                    iDominate[j].add(i);
                    dominateMe[i]++;
                }
            }
        }
        
        for (int i = 0; i < solutionSet.size(); ++ i) {
            if (dominateMe[i] == 0) {
                front[0].add(i);
                solutionSet.get(i).setRank(0);
            }
        }
        
        int i = 0;
        
        int      index;
        Iterator it1;
        while (front[i].size() != 0) {
            ++ i;
            it1 = front[i - 1].iterator();
            
            while (it1.hasNext()) {
                for (Object o : iDominate[(Integer)it1.next()]) {
                    index = (Integer)o;
                    dominateMe[index]--;
                    if (dominateMe[index] == 0) {
                        front[i].add(index);
                        solutionSet.get(index).setRank(i);
                    }
                }
            }
        }
        
        this.ranking_ = new SolutionSet[i];
        
        for (index = 0; index < i; ++ index) {
            this.ranking_[index] = new SolutionSet(front[index].size());
            it1                  = front[index].iterator();
            
            while (it1.hasNext()) {
                this.ranking_[index].add(solutionSet.get((Integer)it1.next()));
            }
        }
    }
    
    public SolutionSet getSubfront(int rank) { return this.ranking_[rank]; }
    
    private static int constraintViolationCompare(Object o1, Object o2) {
        double overall1 = ((Solution)o1).getOverallConstraintViolation();
        double overall2 = ((Solution)o2).getOverallConstraintViolation();
        if (overall1 < 0.0D && overall2 < 0.0D) {
            if (overall1 > overall2) {
                return - 1;
            }
            else {
                return overall2 > overall1 ? 1 : 0;
            }
        }
        else if (overall1 == 0.0D && overall2 < 0.0D) {
            return - 1;
        }
        else {
            return overall1 < 0.0D && overall2 == 0.0D ? 1 : 0;
        }
    }
    
    private static boolean constraintViolationNeedToCompare(Solution s1, Solution s2) {
        return s1.getOverallConstraintViolation() < 0.0D || s2.getOverallConstraintViolation() < 0.0D;
    }
    
    public static int dominanceCompare(Object object1, Object object2) {
        if (object1 == null) {
            return 1;
        }
        else if (object2 == null) {
            return - 1;
        }
        else {
            Solution solution1 = (Solution)object1;
            Solution solution2 = (Solution)object2;
            boolean  dominate1 = false;
            boolean  dominate2 = false;
            if (constraintViolationNeedToCompare(solution1, solution2)) {
                return constraintViolationCompare(solution1, solution2);
            }
            else {
                for (int i = 0; i < solution1.getNumberOfObjectives(); ++ i) {
                    double value1 = solution1.getObjective(i);
                    double value2 = solution2.getObjective(i);
                    byte   flag   = (byte)Double.compare(value1, value2);
                    
                    if (flag == - 1) {
                        dominate1 = true;
                    }
                    
                    if (flag == 1) {
                        dominate2 = true;
                    }
                }
                
                if (dominate1 == dominate2) {
                    return 0;
                }
                else if (dominate1) {
                    return - 1;
                }
                else {
                    return 1;
                }
            }
        }
    }
    
}
