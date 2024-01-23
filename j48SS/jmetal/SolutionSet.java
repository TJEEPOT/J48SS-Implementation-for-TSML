package weka.classifiers.trees.j48SS.jmetal;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;


public class SolutionSet implements Serializable {
    public final  List<Solution> solutionsList_ = new ArrayList<>();
    private final int            m_capacity_;
    
    public SolutionSet(int maximumSize) {
        this.m_capacity_ = maximumSize;
    }
    
    public boolean add(Solution solution) {
        if (this.solutionsList_.size() == this.m_capacity_) {
            return false;
        }
        else {
            this.solutionsList_.add(solution);
            return true;
        }
    }
    
    public Solution get(int i) {
        if (i >= this.solutionsList_.size()) {
            throw new IndexOutOfBoundsException("Index out of Bound " + i);
        }
        else {
            return this.solutionsList_.get(i);
        }
    }
    
    public void sort(Comparator comparator) {
        if (comparator == null) {
            System.err.println("No criterium for comparing exist");
        }
        this.solutionsList_.sort(comparator);
        
    }
    
    public int size() {
        return this.solutionsList_.size();
    }
    
    public void clear() {
        this.solutionsList_.clear();
    }
    
    public void remove(int i) {
        if (i > this.solutionsList_.size() - 1) {
            System.err.println("Size is: " + this.size());
        }
        this.solutionsList_.remove(i);
    }
    
    public SolutionSet union(SolutionSet solutionSet) {
        int newSize = this.size() + solutionSet.size();
        if (newSize < this.m_capacity_) {
            newSize = this.m_capacity_;
        }
        
        SolutionSet union = new SolutionSet(newSize);
        
        int i;
        for (i = 0; i < this.size(); ++ i) {
            union.add(this.get(i));
        }
        
        for (i = this.size(); i < this.size() + solutionSet.size(); ++ i) {
            union.add(solutionSet.get(i - this.size()));
        }
        return union;
    }
}
