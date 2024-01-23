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


package weka.classifiers.trees.j48SS.jmetal;

import java.util.*;

/**
 *  Implementation of NSGA-II.
 *  This implementation of NSGA-II is based on the implementation in JMetal, which was used in the paper:
 *     A.J. Nebro, J.J. Durillo, C.A. Coello Coello, F. Luna, E. Alba
 *     "A Study of Convergence Speed in Multi-Objective Metaheuristics."
 *     Presented in: PPSN'08. Dortmund. September 2008.
 */

public class PNSGAII {
    
    private final MultithreadedEvaluator m_evaluator;
    private final BestShapeletProblem    m_problem;
    
    // Options
    private final BestShapeletCrossover m_crossover;
    private final BestShapeletMutation m_mutation;
    private final BinaryTournament2    m_selection;
    
    // Input parameters
    private final int populationSize;
    private final int maxEvaluations;
    
    // Output Parameters
//    private int evaluations_;

    
    public PNSGAII(BestShapeletProblem problem, MultithreadedEvaluator m_evaluator, int populationSize,
            int maxEvaluations, BestShapeletCrossover m_crossover, BestShapeletMutation m_mutation,
            BinaryTournament2 m_selection){
        this.m_problem      = problem ;
        this.m_evaluator    = m_evaluator;
        this.populationSize = populationSize;
        this.maxEvaluations = maxEvaluations;
        this.m_crossover    = m_crossover;
        this.m_mutation     = m_mutation;
        this.m_selection    = m_selection;
    }
    
    /**
     * Runs the NSGA-II algorithm.
     * @return a <code>SolutionSet</code> that is a set of non dominated solutions
     * as a result of the algorithm execution
     * @throws Exception for issues with mutation and crossover execution
     */
    public SolutionSet execute() throws Exception {
        SolutionSet population;
        SolutionSet offspringPopulation;
        SolutionSet union;
        
        Distance distance = new Distance();
        
        m_evaluator.startEvaluator(m_problem) ;
        
        //Initialize the variables
        population = new SolutionSet(populationSize);
        int evaluations = 0;
        
        // Create the initial solutionSet
        for (int i = 0; i < populationSize; i++) {
            Solution newSolution = new Solution(m_problem);
            m_evaluator.addSolutionForEvaluation(newSolution) ;
        }
        
        List<Solution> solutionList = m_evaluator.parallelEvaluation() ;
        for (Solution solution : solutionList) {
            population.add(solution) ;
            evaluations ++ ;
        }
        
        // Generations
        while (evaluations < maxEvaluations) {
            // Create the offSpring solutionSet
            offspringPopulation = new SolutionSet(populationSize);
            Solution[] parents = new Solution[2];
            for (int i = 0; i < (populationSize / 2); i++) {
                if (evaluations < maxEvaluations) {
                    //obtain parents
                    parents[0] = (Solution) m_selection.execute(population);
                    parents[1] = (Solution) m_selection.execute(population);
                    Solution[] offSpring = (Solution[]) m_crossover.execute(parents);
                    m_mutation.execute(offSpring[0]);
                    m_mutation.execute(offSpring[1]);
                    m_evaluator.addSolutionForEvaluation(offSpring[0]) ;
                    m_evaluator.addSolutionForEvaluation(offSpring[1]) ;
                } // if
            } // for
            
            List<Solution> solutions = m_evaluator.parallelEvaluation() ;
            
            for(Solution solution : solutions) {
                offspringPopulation.add(solution);
                evaluations++;
            }
            
            // Create the solutionSet union of solutionSet and offSpring
            union = population.union(offspringPopulation);
            
            // Ranking the union
            Ranking ranking = new Ranking(union);
            
            int remain = populationSize;
            int index = 0;
            population.clear();
            
            // Obtain the next front
            SolutionSet front = ranking.getSubfront(index);
            
            while ((remain > 0) && (remain >= front.size())) {
                //Assign crowding distance to individuals
                distance.crowdingDistanceAssignment(front, m_problem.getNumberOfObjectives());
                //Add the individuals of this front
                for (int k = 0; k < front.size(); k++) {
                    population.add(front.get(k));
                } // for
                
                //Decrement remain
                remain = remain - front.size();
                
                //Obtain the next front
                index++;
                if (remain > 0) {
                    front = ranking.getSubfront(index);
                } // if
            } // while
            
            // Remain is less than front(index).size, insert only the best one
            if (remain > 0) {  // front contains individuals to insert
                distance.crowdingDistanceAssignment(front, m_problem.getNumberOfObjectives());
                front.sort(new CrowdingComparator());
                for (int k = 0; k < remain; k++) {
                    population.add(front.get(k));
                } // for
            } // if
        } // while
        
        m_evaluator.stopEvaluator();
        
        // Return the first non-dominated front
        Ranking ranking = new Ranking(population);
        return ranking.getSubfront(0);
    } // execute

    private static class CrowdingComparator implements Comparator {
        
        public int compare(Object o1, Object o2) {
            if (o1 == null) {
                return 1;
            } else if (o2 == null) {
                return -1;
            } else {
                int flagComparatorRank = rankCompare(o1, o2);
                if (flagComparatorRank != 0) {
                    return flagComparatorRank;
                } else {
                    double distance1 = ((Solution)o1).getCrowdingDistance();
                    double distance2 = ((Solution)o2).getCrowdingDistance();
                    if (distance1 > distance2) {
                        return -1;
                    } else {
                        return distance1 < distance2 ? 1 : 0;
                    }
                }
            }
        }
        
        public int rankCompare(Object o1, Object o2) {
            if (o1 == null) {
                return 1;
            } else if (o2 == null) {
                return -1;
            } else {
                Solution solution1 = (Solution)o1;
                Solution solution2 = (Solution)o2;
                if (solution1.getRank() < solution2.getRank()) {
                    return -1;
                } else {
                    return solution1.getRank() > solution2.getRank() ? 1 : 0;
                }
            }
        }
    }
} // pNSGAII