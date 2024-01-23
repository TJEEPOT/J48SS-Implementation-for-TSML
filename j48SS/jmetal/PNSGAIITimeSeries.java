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

import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

/**
 * Class to configure and execute the pNSGA-II algorithm. pNSGA-II is a multithreaded version of NSGA-II, where
 * evaluations are carried out in parallel. This version has been modified to extract time series shapelets
 * for use with J48SS.
 */
public class PNSGAIITimeSeries {
    
    public static Map<String, double[]> getGeneticSolution(
            double minSplit, List<String> listOfInstancesAll,
            int populationSize, int maxEvaluations, double crossoverProbability, double mutationProbability,
            double weight, boolean isVerbose) throws Exception {
        
        BestShapeletProblem problem = new BestShapeletProblem(listOfInstancesAll, minSplit,
                isVerbose);
        BestShapeletCrossover  crossover              = new BestShapeletCrossover(crossoverProbability);
        BestShapeletMutation   mutation               = new BestShapeletMutation(mutationProbability);
        BinaryTournament2      selection              = new BinaryTournament2(null);
        int                    threads                = 0; // use all available threads
        MultithreadedEvaluator multithreadedEvaluator = new MultithreadedEvaluator(threads);
        
        PNSGAII algorithm = new PNSGAII(problem, multithreadedEvaluator, populationSize, maxEvaluations, crossover,
                mutation, selection);
        SolutionSet population = algorithm.execute();
        return getShapeletFromPopulation(population, weight, isVerbose);
    }
    
    private static Map<String, double[]> getShapeletFromPopulation(
            SolutionSet population, double weight, boolean isVerbose) {
        // We want to minimize: (1-w)*s_length + w*(s_IG) : s_IG is negative from the EA)
        double bestValueScore = Double.MAX_VALUE;
        
        double   bestValueCompressionRatio = - 1.0D;
        double[] nullIGShapelet            = null;
        double   minIG                     = Double.MAX_VALUE;
        double   maxComprRatio             = - 1.0D;
        
        for (Solution sol : population.solutionsList_) {
            String[] negIGandComprString = sol.toString().split(" ");
            double   negIG               = Double.parseDouble(negIGandComprString[0]);
            double   comprRatio          = Double.parseDouble(negIGandComprString[1]);
            if (comprRatio > maxComprRatio) {
                maxComprRatio = comprRatio;
            }
            
            if (negIG < minIG) {
                minIG = negIG;
            }
        }
        
        minIG *= - 1.0D; // multiply minIG by -1 so it does not change the sign in the weighted calculation
        double[]           bestShapelet   = null;
        double[]           bestShapeletIG = new double[] {- 1.0D};
        Solution           sol;
        double             comprRatio;
        Iterator<Solution> iter          = population.solutionsList_.iterator();
        
        // TODO: This needs untangling again -MS
        while (true) {
            double   curValue;
            String[] shapeletString;
            double[] shapeletDouble;
            int      i;
            double   negIG;
            do {
                if (! iter.hasNext()) {
                    if (bestValueCompressionRatio == - 1.0D) {
                        bestShapeletIG[0]         = 0.0D;
                        bestValueCompressionRatio = 0.0D;
                        bestShapelet              = nullIGShapelet;
                    }
                    
                    Map<String, double[]> returnMap = new HashMap<>();
                    if (isVerbose) {
                        System.out.println("bestShapeletIG: " + bestShapeletIG[0] +
                                " bestShapeletCompressionRatio: " + bestValueCompressionRatio);
                    }
                    returnMap.put("shapelet", bestShapelet);
                    returnMap.put("IG", bestShapeletIG);
                    if (bestShapeletIG[0] == - 1.0D) {
                        System.err.println("Error in extracting shapelet");
                        System.exit(1);
                    }
                    return returnMap;
                }
                
                sol = iter.next();
                String[] negIGandComprString = sol.toString().split(" ");
                negIG      = Double.parseDouble(negIGandComprString[0]);
                comprRatio = Double.parseDouble(negIGandComprString[1]);
                double negIGREL = negIG / minIG;
                if (Double.isNaN(negIGREL)) {
                    negIGREL = 0.0D;
                }
                
                double comprRatioREL = comprRatio / maxComprRatio;
                if (Double.isNaN(comprRatioREL)) {
                    comprRatioREL = 0.0D;
                }
                
                curValue = (1.0D - weight) * comprRatioREL + weight * (1.0D - negIGREL * - 1.0D);
                if (negIGREL == 0.0D || comprRatioREL == 0.0D) {
                    curValue = Double.MAX_VALUE;
                    if (nullIGShapelet == null) {
                        shapeletString = sol.getDecisionVariables()[0].toString().split(",");
                        shapeletDouble = new double[shapeletString.length];
                        
                        for (i = 0; i < shapeletDouble.length; ++ i) {
                            shapeletDouble[i] = Double.parseDouble(shapeletString[i]);
                        }
                        nullIGShapelet = shapeletDouble;
                    }
                }
            }while (! (curValue < bestValueScore) &&
                    (curValue != bestValueScore || ! (comprRatio < bestValueCompressionRatio)));
            
            shapeletString = sol.getDecisionVariables()[0].toString().split(",");
            shapeletDouble = new double[shapeletString.length];
            
            for (i = 0; i < shapeletDouble.length; ++ i) {
                shapeletDouble[i] = Double.parseDouble(shapeletString[i]);
            }
            
            bestShapelet              = shapeletDouble;
            bestShapeletIG[0]         = - negIG;
            bestValueScore            = curValue;
            bestValueCompressionRatio = comprRatio;
        }
    }
}
