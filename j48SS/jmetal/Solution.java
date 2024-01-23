package weka.classifiers.trees.j48SS.jmetal;

import java.io.Serializable;

public class Solution implements Serializable {
    private final BestShapeletSolutionType type_;
    private final BestShapelet[]           variable_;
    private final double[]                 objective_;
    private final int      numberOfObjectives_;
    private       int      rank_;
    private double overallConstraintViolation_;
    private double crowdingDistance_ = 0.0D;
    
    public Solution(BestShapeletProblem problem) {
        this.type_ = problem.getSolutionType();
        this.numberOfObjectives_ = problem.getNumberOfObjectives();
        this.objective_ = new double[this.numberOfObjectives_];
        this.variable_ = this.type_.createVariables();
    }
    
    public Solution(Solution solution) {
        this.type_ = solution.type_;
        this.numberOfObjectives_ = solution.getNumberOfObjectives();
        this.objective_ = new double[this.numberOfObjectives_];
        
        for(int i = 0; i < this.objective_.length; ++i) {
            this.objective_[i] = solution.getObjective(i);
        }
    
        BestShapelet[] variables = new BestShapelet[solution.variable_.length];
        for(int i = 0; i < solution.variable_.length; ++i) {
            variables[i] = solution.variable_[i].deepCopy();
        }
        this.variable_ = variables;
        this.overallConstraintViolation_ = solution.overallConstraintViolation_;
        this.crowdingDistance_ = solution.crowdingDistance_;
        this.rank_ = solution.rank_;
    }
    
    public void setCrowdingDistance(double distance) { this.crowdingDistance_ = distance;
    }
    
    public double getCrowdingDistance() { return this.crowdingDistance_; }
    
    public void setObjective(int i, double value) { this.objective_[i] = value; }
    
    public double getObjective(int i) { return this.objective_[i]; }
    
    public int getNumberOfObjectives() { return this.objective_ == null ? 0 : this.numberOfObjectives_; }
    
    public String toString() {
        StringBuilder aux = new StringBuilder();
        
        for(int i = 0; i < this.numberOfObjectives_; ++i) {
            aux.append(this.getObjective(i)).append(" ");
        }
        return aux.toString();
    }
    
    public BestShapelet[] getDecisionVariables() { return this.variable_; }
    
    public void setRank(int value) { this.rank_ = value; }
    
    public int getRank() { return this.rank_; }
    
    public double getOverallConstraintViolation() { return this.overallConstraintViolation_; }
    
    public BestShapeletSolutionType getType() { return this.type_; }

}
