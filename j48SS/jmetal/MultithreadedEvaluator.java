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

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Vector;
import java.util.concurrent.*;

/**
 * @author Antonio J. Nebro
 * Class for evaluating solutions in parallel using threads
 */
public class MultithreadedEvaluator {
    private int numberOfThreads_ ;
    private BestShapeletProblem problem_ ;
    private ExecutorService executor_ ;
    private Collection<Callable<Solution>> taskList_ ;
    
    /**
     * @author Antonio J. Nebro
     * Private class representing tasks to evaluate solutions.
     */
    
    private static class EvaluationTask implements Callable<Solution> {
        private final BestShapeletProblem problem_ ;
        private final Solution            solution_ ;
        
        /**
         * Constructor
         * @param problem Problem to solve
         * @param solution Solution to evaluate
         */
        public EvaluationTask(BestShapeletProblem problem,  Solution solution) {
            problem_ = problem ;
            solution_ = solution ;
        }
        
        public Solution call() {
            problem_.evaluate(solution_) ;
            return solution_ ;
        }
    }
    
    /**
     * Constructor
     * @param threads number of threads for the evaluator to use.
     */
    public MultithreadedEvaluator(int threads) {
        numberOfThreads_ = threads ;
        if (threads < 1) {
            numberOfThreads_ = Runtime.getRuntime().availableProcessors();
        }
    }
    
    /**
     * Constructor
     * @param problem problem to solve.
     */
    public void startEvaluator(BestShapeletProblem problem) {
        executor_ = Executors.newFixedThreadPool(numberOfThreads_) ;
//        System.out.println("Cores: "+ numberOfThreads_) ;
        taskList_ = null ;
        problem_ = problem ;
    }
    
    /**
     * Adds a solution to be evaluated to a list of tasks
     * @param solution Solution to be evaluated
     */
    public void addSolutionForEvaluation(Solution solution) {
        if (taskList_ == null)
            taskList_ = new ArrayList<>();
        
        taskList_.add(new EvaluationTask(problem_, solution)) ;
    }
    
    /**
     * Evaluates a list of solutions
     * @return A list with the evaluated solutions
     */
    public List<Solution> parallelEvaluation() {
        List<Future<Solution>> future = null ;
        try {
            // Spin up a number of threads to work on the tasks in tasklist_. This eventually calls
            // BestShapeletProblem.evaluate() on each problem in each EvaluationTask.
            future = executor_.invokeAll(taskList_);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        List<Solution> solutionList = new Vector<>() ;
    
        assert future != null;
        for(Future<Solution> result : future){
            Solution solution = null ;
            try {
                solution = result.get();
                solutionList.add(solution) ;
            } catch (InterruptedException | ExecutionException e) {
                e.printStackTrace();
            }
        }
        taskList_ = null ;
        return solutionList ;
    }
    
    /**
     * Shutdown the executor
     */
    public void stopEvaluator() {
        executor_.shutdown() ;
    }
}
