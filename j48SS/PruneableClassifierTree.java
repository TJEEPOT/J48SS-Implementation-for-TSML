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

import java.util.Random;
import weka.core.Capabilities;
import weka.core.Instances;
import weka.core.RevisionUtils;
import weka.core.Utils;
import weka.core.Capabilities.Capability;

/**
 * Class for handling a tree structure that can be pruned using a pruning set.
 *
 * @author Eibe Frank (eibe@cs.waikato.ac.nz)
 * @version $Revision: 8984 $
 */
public class PruneableClassifierTree extends ClassifierTree {
    /** for serialization */
    static final long serialVersionUID = -555775736857600201L;
    
    /** True if the tree is to be pruned. */
    protected boolean m_pruneTheTree;
    
    /** How many subsets of equal size? One used for pruning, the rest for training. */
    protected int m_numSets;
    
    /** Cleanup after the tree has been built. */
    protected boolean m_cleanup;
    
    /** The random number seed. */
    protected int m_seed;
    
    /**
     * Constructor for pruneable tree structure. Stores reference
     * to associated training data at each node.
     *
     * @param toSelectLocModel selection method for local splitting model
     * @param pruneTree true if the tree is to be pruned
     * @param num number of subsets of equal size
     * @param cleanup true if cleanup should be performed after the tree has been built
     * @param seed the seed value to use
     */
    public PruneableClassifierTree(ModelSelection toSelectLocModel, boolean pruneTree, int num, boolean cleanup,
            int seed) {
        super(toSelectLocModel);
        this.m_pruneTheTree = pruneTree;
        this.m_numSets      = num;
        this.m_cleanup    = cleanup;
        this.m_seed = seed;
    }
    
    /**
     * Returns default capabilities of the classifier tree.
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
     * @param data the data to build the tree from
     * @throws Exception if tree can't be built successfully
     */
    public void buildClassifier(Instances data) throws Exception {
        // can classifier tree handle the data?
        this.getCapabilities().testWithFail(data);
    
        // remove instances with missing class
        data = new Instances(data);
        data.deleteWithMissingClass();
        
        Random random = new Random(m_seed);
        data.stratify(m_numSets);
        this.buildTree(data.trainCV(m_numSets, m_numSets - 1, random),
                data.testCV(m_numSets, m_numSets - 1), !m_cleanup);
        if (m_pruneTheTree) {
            this.prune();
        }
        
        if (m_cleanup) {
            this.cleanup(new Instances(data, 0));
        }
        
    }
    
    /**
     * Prunes a tree.
     *
     * @throws Exception if tree can't be pruned successfully
     */
    public void prune() throws Exception {
        if (!m_isLeaf) {
            // Prune all subtrees.
            for(int i = 0; i < m_sons.length; i++) {
                this.son(i).prune();
            }
    
            // Decide if leaf is best choice.
            if (Utils.smOrEq(this.errorsForLeaf(), this.errorsForTree())) {
                // Free son Trees
                m_sons = null;
                m_isLeaf = true;
                
                // Get NoSplit Model for node.
                m_localModel = new NoSplit(this.localModel().distribution());
            }
        }
        
    }
    
    /**
     * Returns a newly created tree.
     *
     * @param train the training data
     * @param test the test data
     * @return the generated tree
     * @throws Exception if something goes wrong
     */
    protected ClassifierTree getNewTree(Instances train, Instances test) throws Exception {
        PruneableClassifierTree newTree =
                new PruneableClassifierTree(m_toSelectModel, m_pruneTheTree, m_numSets, m_cleanup, m_seed);
        newTree.buildTree(train, test, !m_cleanup);
        return newTree;
    }
    
    /**
     * Returns the estimated errors for tree.
     */
    private double errorsForTree(){
        double errors = 0.0D;
        if (m_isLeaf) {
            return this.errorsForLeaf();
        }
        
        for(int i = 0; i < m_sons.length; i++) {
            if (Utils.eq(this.localModel().distribution().perBag(i), 0.0D)) {
                errors += m_test.perBag(i) - m_test.perClassPerBag(i, this.localModel().distribution().maxClass());
            } else {
                errors += this.son(i).errorsForTree();
            }
        }
        return errors;

    }
    
    /**
     * Returns the estimated errors for leaf.
     */
    private double errorsForLeaf() {
        return m_test.total() - m_test.perClass(this.localModel().distribution().maxClass());
    }
    
    /**
     * Method just exists to make program easier to read.
     */
    private ClassifierSplitModel localModel() {
        return m_localModel;
    }
    
    /**
     * Method just exists to make program easier to read.
     */
    private PruneableClassifierTree son(int index) {
        return (PruneableClassifierTree)m_sons[index];
    }
    
    /**
     * Returns the revision string.
     *
     * @return		the revision
     */
    public String getRevision() { return RevisionUtils.extract("$Revision: 8984 $"); }
}
