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

package weka.classifiers.trees.j48SS;

import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Instances;
import weka.core.RevisionUtils;
import weka.core.Utils;

/**
 * Class for handling a tree structure that can be pruned using a pruning set.
 *
 * @author Eibe Frank (eibe@cs.waikato.ac.nz)
 * @version $Revision: 8984 $
 */
public class C45PruneableClassifierTree extends ClassifierTree {
    /** for serialization */
    static final long serialVersionUID = - 4813820170260388194L;
    
    /** True if the tree is to be pruned. */
    protected boolean m_pruneTheTree;
    
    /** True if the tree is to be collapsed. */
    protected boolean m_collapseTheTree;
    
    /** The confidence factor for pruning. */
    protected float m_CF;
    
    /** Is subtree raising to be performed? */
    protected boolean m_subtreeRaising;
    
    /** Cleanup after the tree has been built. */
    protected boolean m_cleanup;
    
    /**
     * Constructor for prunable tree structure. Stores reference
     * to associated training data at each node.
     *
     * @param toSelectLocModel selection method for local splitting model
     * @param pruneTree        true if the tree is to be pruned
     * @param cf               the confidence factor for pruning
     * @param raiseTree        true if subtree raising to be performed
     * @param cleanup          true if cleanup should be performed after the tree has been built
     */
    public C45PruneableClassifierTree(
            ModelSelection toSelectLocModel, boolean pruneTree, float cf,
            boolean raiseTree, boolean cleanup, boolean collapseTree) {
        super(toSelectLocModel);
        m_pruneTheTree    = pruneTree;
        m_CF              = cf;
        m_subtreeRaising  = raiseTree;
        m_cleanup         = cleanup;
        m_collapseTheTree = collapseTree;
    }
    
    /**
     * Returns default capabilities of the classifier tree.
     *
     * @return the capabilities of this classifier tree
     */
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();
        
        // attributes
        result.enable(Capability.NOMINAL_ATTRIBUTES);
        result.enable(Capability.NUMERIC_ATTRIBUTES);
        result.enable(Capability.DATE_ATTRIBUTES);
        result.enable(Capability.MISSING_VALUES);
        result.enable(Capability.STRING_ATTRIBUTES);
        
        // class
        result.enable(Capability.NOMINAL_CLASS);
        result.enable(Capability.MISSING_CLASS_VALUES);
        
        // instances
        result.setMinimumNumberInstances(0);
        return result;
    }
    
    /**
     * Method for building a pruneable classifier tree.
     *
     * @param data the data for building the tree
     *
     * @throws Exception if something goes wrong
     */
    public void buildClassifier(Instances data) throws Exception {
        this.getCapabilities().testWithFail(data);
        data = new Instances(data);
        data.deleteWithMissingClass();
        this.buildTree(data, m_subtreeRaising || ! m_cleanup);
        if (m_collapseTheTree) {
            this.collapse();
        }
        
        if (m_pruneTheTree) {
            this.prune();
        }
        
        if (m_cleanup) {
            this.cleanup(new Instances(data, 0));
        }
        
    }
    
    /**
     * Collapses a tree to a node if training error doesn't increase.
     */
    public final void collapse() {
        if (! m_isLeaf) {
            double errorsOfSubtree = this.getTrainingErrors();
            double errorsOfTree    = m_localModel.distribution().numIncorrect();
            if (errorsOfSubtree >= errorsOfTree - 0.001D) {
                m_sons       = null;
                m_isLeaf     = true;
                m_localModel = new NoSplit(m_localModel.distribution());
            }
            else {
                for (ClassifierTree m_son : m_sons) {
                    ((C45PruneableClassifierTree)m_son).collapse();
                }
            }
        }
    }
    
    /**
     * Prunes a tree using C4.5's pruning procedure.
     *
     * @throws Exception for getEstimatedErrorsForBranch()
     */
    public void prune() throws Exception {
        if (! m_isLeaf) {
            // Prune all subtrees.
            for (ClassifierTree m_son : m_sons) {
                ((C45PruneableClassifierTree)m_son).prune();
            }
            
            // Compute error for largest branch
            int    indexOfLargestBranch = m_localModel.distribution().maxBag();
            double errorsLargestBranch;
            if (m_subtreeRaising) {
                errorsLargestBranch = ((C45PruneableClassifierTree)m_sons[indexOfLargestBranch])
                        .getEstimatedErrorsForBranch(m_train);
            }
            else {
                errorsLargestBranch = Double.MAX_VALUE;
            }
            
            // Compute error if this Tree would be leaf
            double errorsLeaf = this.getEstimatedErrorsForDistribution(m_localModel.distribution());
            
            // Compute error for the whole subtree
            double errorsTree = this.getEstimatedErrors();
            
            // Decide if leaf is best choice.
            if (Utils.smOrEq(errorsLeaf, errorsTree + 0.1D) && Utils.smOrEq(errorsLeaf, errorsLargestBranch + 0.1D)) {
                // Free son Trees
                m_sons   = null;
                m_isLeaf = true;
                
                // Get NoSplit Model for node.
                m_localModel = new NoSplit(m_localModel.distribution());
                return;
            }
            
            // Decide if largest branch is better choice than whole subtree.
            if (Utils.smOrEq(errorsLargestBranch, errorsTree + 0.1D) && ! m_sons[indexOfLargestBranch].m_isLeaf) {
                C45PruneableClassifierTree largestBranch = ((C45PruneableClassifierTree)m_sons[indexOfLargestBranch]);
                m_sons       = largestBranch.m_sons;
                m_localModel = largestBranch.m_localModel;
                m_isLeaf     = largestBranch.m_isLeaf;
                this.newDistribution(m_train);
                this.prune();
            }
        }
    }
    
    /**
     * Returns a newly created tree.
     *
     * @param data the data to work with
     *
     * @return the new tree
     * @throws Exception from buildTree()
     */
    protected ClassifierTree getNewTree(Instances data) throws Exception {
        C45PruneableClassifierTree newTree = new C45PruneableClassifierTree(m_toSelectModel, m_pruneTheTree,
                m_CF, m_subtreeRaising, m_cleanup, m_collapseTheTree);
        newTree.buildTree(data, m_subtreeRaising || ! m_cleanup);
        return newTree;
    }
    
    /**
     * Computes estimated errors for tree.
     *
     * @return the estimated errors
     */
    private double getEstimatedErrors() {
        double errors = 0.0D;
        if (m_isLeaf) {
            return this.getEstimatedErrorsForDistribution(m_localModel.distribution());
        }
        else {
            for (ClassifierTree m_son : m_sons) {
                errors += ((C45PruneableClassifierTree)m_son).getEstimatedErrors();
            }
            return errors;
        }
    }
    
    /**
     * Computes estimated errors for one branch.
     *
     * @param data the data to work with
     *
     * @return the estimated errors
     * @throws Exception if something goes wrong
     */
    private double getEstimatedErrorsForBranch(Instances data) throws Exception {
        double errors = 0.0D;
        if (m_isLeaf) {
            return this.getEstimatedErrorsForDistribution(new Distribution(data));
        }
        else {
            Distribution savedDist = m_localModel.m_distribution;
            m_localModel.resetDistribution(data);
            Instances[] localInstances = m_localModel.split(data);
            m_localModel.m_distribution = savedDist;
            
            for (int i = 0; i < m_sons.length; i++) {
                errors += ((C45PruneableClassifierTree)m_sons[i]).getEstimatedErrorsForBranch(localInstances[i]);
            }
            return errors;
        }
    }
    
    /**
     * Computes estimated errors for leaf.
     *
     * @param theDistribution the distribution to use
     *
     * @return the estimated errors
     */
    private double getEstimatedErrorsForDistribution(Distribution theDistribution) {
        return Utils.eq(theDistribution.total(), 0.0D) ? 0.0D : theDistribution.numIncorrect() + Stats
                .addErrs(theDistribution.total(), theDistribution.numIncorrect(), m_CF);
    }
    
    /**
     * Computes errors of tree on training data.
     *
     * @return the training errors
     */
    private double getTrainingErrors() {
        double errors = 0.0D;
        if (m_isLeaf) {
            return m_localModel.distribution().numIncorrect();
        }
        else {
            for (ClassifierTree m_son : m_sons) {
                errors += ((C45PruneableClassifierTree)m_son).getTrainingErrors();
            }
            return errors;
        }
    }
    
    /**
     * Computes new distributions of instances for nodes
     * in tree.
     *
     * @param data the data to compute the distributions for
     *
     * @throws Exception if something goes wrong
     */
    private void newDistribution(Instances data) throws Exception {
        m_localModel.resetDistribution(data);
        m_train = data;
        if (! m_isLeaf) {
            Instances[] localInstances = m_localModel.split(data);
            
            for (int i = 0; i < m_sons.length; i++) {
                ((C45PruneableClassifierTree)m_sons[i]).newDistribution(localInstances[i]);
            }
        }
        // Check whether there are some instances at the leaf now!
        else if (! Utils.eq(data.sumOfWeights(), 0.0D)) {
            m_isEmpty = false;
        }
    }
    
    /**
     * Returns the revision string.
     */
    public String getRevision() {
        return RevisionUtils.extract("$Revision: 11006 $");
    }
}
