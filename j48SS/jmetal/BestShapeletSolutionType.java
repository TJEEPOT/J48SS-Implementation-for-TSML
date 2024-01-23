package weka.classifiers.trees.j48SS.jmetal;

import java.util.Map;

// MS: may be able to move this...
public class BestShapeletSolutionType {
    Map<Integer, double[]> mapOfInstances;
    int maxInstanceLength;
    double maxInstanceValue;
    public BestShapeletProblem problem_;
    
    public BestShapeletSolutionType(BestShapeletProblem problem, Map<Integer, double[]> mapOfInstances_in, int maxInstanceLength_in, double maxInstanceValue_in) {
        this.problem_ = problem;
        this.mapOfInstances = mapOfInstances_in;
        this.maxInstanceLength = maxInstanceLength_in;
        this.maxInstanceValue = maxInstanceValue_in;
    }
    
    public BestShapelet[] createVariables() {
        return new BestShapelet[]{new BestShapelet(this.mapOfInstances, this.maxInstanceLength,
                this.maxInstanceValue)};
    }
}
