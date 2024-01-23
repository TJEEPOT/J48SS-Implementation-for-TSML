package weka.classifiers.trees.j48SS.jmetal;

import java.util.Comparator;

public class Distance {
    public Distance() {
    }
    
    public double[][] distanceMatrix(SolutionSet solutionSet) {
        double[][] distance = new double[solutionSet.size()][solutionSet.size()];
        
        for(int i = 0; i < solutionSet.size(); ++i) {
            distance[i][i] = 0.0D;
            Solution solutionI = solutionSet.get(i);
            
            for(int j = i + 1; j < solutionSet.size(); ++j) {
                Solution solutionJ = solutionSet.get(j);
                distance[i][j] = this.distanceBetweenObjectives(solutionI, solutionJ);
                distance[j][i] = distance[i][j];
            }
        }
        
        return distance;
    }
    
    public double distanceBetweenObjectives(Solution solutionI, Solution solutionJ) {
        double distance = 0.0D;
        
        for(int nObj = 0; nObj < solutionI.getNumberOfObjectives(); ++nObj) {
            double diff = solutionI.getObjective(nObj) - solutionJ.getObjective(nObj);
            distance += Math.pow(diff, 2.0D);
        }
        
        return Math.sqrt(distance);
    }
    
    public void crowdingDistanceAssignment(SolutionSet solutionSet, int nObjs) {
        int size = solutionSet.size();
        if (size != 0) {
            if (size == 1) {
                solutionSet.get(0).setCrowdingDistance(1.0D / 0.0);
            } else if (size == 2) {
                solutionSet.get(0).setCrowdingDistance(1.0D / 0.0);
                solutionSet.get(1).setCrowdingDistance(1.0D / 0.0);
            } else {
                SolutionSet front = new SolutionSet(size);

                for(int i = 0; i < size; ++i) {
                    front.add(solutionSet.get(i));
                }
                
                for(int i = 0; i < size; ++i) {
                    front.get(i).setCrowdingDistance(0.0D);
                }
                
                for(int i = 0; i < nObjs; ++i) {
                    front.sort(new ObjectiveComparator(i));
                    double objetiveMinn = front.get(0).getObjective(i);
                    double objetiveMaxn = front.get(front.size() - 1).getObjective(i);
                    front.get(0).setCrowdingDistance(1.0D / 0.0);
                    front.get(size - 1).setCrowdingDistance(1.0D / 0.0);
                    
                    for(int j = 1; j < size - 1; ++j) {
                        double distance = front.get(j + 1).getObjective(i) - front.get(j - 1).getObjective(i);
                        distance /= objetiveMaxn - objetiveMinn;
                        distance += front.get(j).getCrowdingDistance();
                        front.get(j).setCrowdingDistance(distance);
                    }
                }
            }
        }
    }
    
    private static class ObjectiveComparator implements Comparator {
        private final int     nObj;
        private final boolean ascendingOrder_;
        
        public ObjectiveComparator(int nObj) {
            this.nObj = nObj;
            this.ascendingOrder_ = true;
        }
        
        public int compare(Object o1, Object o2) {
            if (o1 == null) {
                return 1;
            } else if (o2 == null) {
                return -1;
            } else {
                double objetive1 = ((Solution)o1).getObjective(this.nObj);
                double objetive2 = ((Solution)o2).getObjective(this.nObj);
                if (this.ascendingOrder_) {
                    if (objetive1 < objetive2) {
                        return -1;
                    } else {
                        return objetive1 > objetive2 ? 1 : 0;
                    }
                } else if (objetive1 < objetive2) {
                    return 1;
                } else {
                    return objetive1 > objetive2 ? -1 : 0;
                }
            }
        }
    }
}
