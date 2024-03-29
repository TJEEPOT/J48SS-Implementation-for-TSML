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

import weka.core.*;

import java.io.Serializable;
import java.util.LinkedList;
import java.util.Queue;

/**
 * Class for handling a tree structure used for
 * classification.
 *
 * @author Eibe Frank (eibe@cs.waikato.ac.nz)
 * @version $Revision: 9117 $
 */
public class ClassifierTree implements Drawable, Serializable, CapabilitiesHandler, RevisionHandler {
    /** for serialization */
    static final long serialVersionUID = - 8722249377542734193L;
    
    
    /** The model selection method. */
    protected ModelSelection m_toSelectModel;
    
    /** Local model at node. */
    protected ClassifierSplitModel m_localModel;
    
    /** References to sons. */
    protected ClassifierTree[] m_sons;
    
    /** True if node is leaf. */
    protected boolean m_isLeaf;
    
    /** True if node is empty. */
    protected boolean m_isEmpty;
    
    /** The training instances. */
    protected Instances m_train;
    
    /** The pruning instances. */
    protected Distribution m_test;
    
    /** The id for the node. */
    protected int m_id;
    
    /** For getting a unique ID when outputting the tree (hashcode isn't guaranteed unique) */
    private static long PRINTED_NODES = 0;
    
    /**
     * Gets the next unique node ID.
     *
     * @return the next unique node ID.
     */
    public boolean isLeaf() { return m_isLeaf; }
    
    /**
     * Gets the next unique node ID.
     *
     * @return the next unique node ID.
     */
    protected static long nextID() { return PRINTED_NODES++; }
    
    /**
     * Resets the unique node ID counter (e.g. between repeated separate print types)
     */
    protected static void resetID() { PRINTED_NODES = 0L; }
    
    /**
     * Constructor.
     */
    public ClassifierTree(ModelSelection toSelectLocModel) { m_toSelectModel = toSelectLocModel; }
    
    /**
     * Returns default capabilities of the classifier tree.
     *
     * @return the capabilities of this classifier tree
     */
    public Capabilities getCapabilities() {
        Capabilities result = new Capabilities(this);
        result.enableAll();
        return result;
    }
    
    /**
     * Method for building a classifier tree.
     *
     * @param data the data to build the tree from
     *
     * @throws Exception if something goes wrong
     */
    public void buildClassifier(Instances data) throws Exception {
        // can classifier tree handle the data?
        this.getCapabilities().testWithFail(data);
        
        // remove instances with missing class
        data = new Instances(data);
        data.deleteWithMissingClass();
        
        this.buildTree(data, false);
    }
    
    /**
     * Builds the tree structure.
     *
     * @param data     the data for which the tree structure is to be generated.
     * @param keepData is training data to be kept?
     *
     * @throws Exception if something goes wrong
     */
    public void buildTree(Instances data, boolean keepData) throws Exception {
        if (keepData) {
            m_train = data;
        }
        
        m_test       = null;
        m_isLeaf     = false;
        m_isEmpty    = false;
        m_sons       = null;
        m_localModel = m_toSelectModel.selectModel(data);
        if (m_localModel.numSubsets() > 1) {
            Instances[] localInstances = m_localModel.split(data);
            m_sons = new ClassifierTree[m_localModel.numSubsets()];
            
            for (int i = 0; i < m_sons.length; i++) {
                m_sons[i]    = this.getNewTree(localInstances[i]);
                localInstances[i] = null;
            }
        }
        else {
            m_isLeaf = true;
            if (Utils.eq(data.sumOfWeights(), 0.0D)) {
                m_isEmpty = true;
            }
        }
    }
    
    /**
     * Builds the tree structure with hold out set
     *
     * @param train    the data for which the tree structure is to be generated.
     * @param test     the test data for potential pruning
     * @param keepData is training Data to be kept?
     *
     * @throws Exception if something goes wrong
     */
    public void buildTree(Instances train, Instances test, boolean keepData) throws Exception {
        if (keepData) {
            m_train = train;
        }
        
        m_isLeaf     = false;
        m_isEmpty    = false;
        m_sons       = null;
        m_localModel = m_toSelectModel.selectModel(train, test);
        m_test       = new Distribution(test, m_localModel);
        if (m_localModel.numSubsets() > 1) {
            Instances[] localTrain = m_localModel.split(train);
            Instances[] localTest  = m_localModel.split(test);
            m_sons = new ClassifierTree[m_localModel.numSubsets()];
            
            for (int i = 0; i < m_sons.length; i++) {
                m_sons[i] = this.getNewTree(localTrain[i], localTest[i]);
                localTrain[i]  = null;
                localTest[i]   = null;
            }
        }
        else {
            m_isLeaf = true;
            if (Utils.eq(train.sumOfWeights(), 0.0D)) {
                m_isEmpty = true;
            }
        }
    }
    
    /**
     * Classifies an instance.
     *
     * @param instance the instance to classify
     *
     * @return the classification
     * @throws Exception if something goes wrong
     */
    public double classifyInstance(Instance instance) throws Exception {
        double maxProb  = - 1.0D;
        int    maxIndex = 0;
        
        for (int j = 0; j < instance.numClasses(); j++) {
            double currentProb = this.getProbs(j, instance, 1.0D);
            if (Utils.gr(currentProb, maxProb)) {
                maxIndex = j;
                maxProb  = currentProb;
            }
        }
        
        return maxIndex;
    }
    
    /**
     * Cleanup in order to save memory.
     *
     * @param justHeaderInfo just the header info from the data
     */
    public final void cleanup(Instances justHeaderInfo) {
        m_train = justHeaderInfo;
        m_test  = null;
        if (! m_isLeaf) {
            for (ClassifierTree m_son : m_sons) {
                m_son.cleanup(justHeaderInfo);
            }
        }
    }
    
    /**
     * Returns class probabilities for a weighted instance.
     *
     * @param instance   the instance to get the distribution for
     * @param useLaplace whether to use laplace or not
     *
     * @return the distribution
     * @throws Exception if something goes wrong
     */
    public final double[] distributionForInstance(Instance instance, boolean useLaplace) throws Exception {
        double[] doubles = new double[instance.numClasses()];
        
        for (int i = 0; i < doubles.length; i++) {
            if (! useLaplace) {
                doubles[i] = this.getProbs(i, instance, 1.0D);
            }
            else {
                doubles[i] = this.getProbsLaplace(i, instance, 1.0D);
            }
        }
        
        return doubles;
    }
    
    /**
     * Assigns a unique id to every node in the tree.
     *
     * @param lastID the last ID that was assign
     *
     * @return the new current ID
     */
    public int assignIDs(int lastID) {
        int currLastID = lastID + 1;
        m_id = currLastID;
        if (m_sons != null) {
            for (ClassifierTree m_son : m_sons) {
                currLastID = m_son.assignIDs(currLastID);
            }
        }
        return currLastID;
    }
    
    /**
     * Returns the type of graph this classifier
     * represents.
     *
     * @return 1
     */
    public int graphType() { return 1; }
    
    /**
     * Returns graph describing the tree.
     *
     * @return the tree as graph
     * @throws Exception if something goes wrong
     */
    public String graph() throws Exception {
        StringBuilder text = new StringBuilder();
        this.assignIDs(- 1);
        text.append("digraph J48Tree {\n");
        if (m_isLeaf) {
            text.append("N").append(m_id).append(" [label=\"")
                    .append(Utils.backQuoteChars(m_localModel.dumpLabel(0, m_train))).append("\" ")
                    .append("shape=box style=filled ");
            if (m_train != null && m_train.numInstances() > 0) {
                text.append("data =\n").append(m_train).append("\n");
                text.append(",\n");
            }
            
            text.append("]\n");
        }
        else {
            text.append("N").append(m_id).append(" [label=\"")
                    .append(Utils.backQuoteChars(m_localModel.leftSide(m_train))).append("\" ");
            if (m_train != null && m_train.numInstances() > 0) {
                text.append("data =\n").append(m_train).append("\n");
                text.append(",\n");
            }
            text.append("]\n");
            this.graphTree(text);
        }
        return text + "}\n";
    }
    
    /**
     * Returns tree in prefix order.
     *
     * @return the prefix order
     * @throws Exception if something goes wrong
     */
    public String prefix() throws Exception {
        StringBuilder text = new StringBuilder();
        if (m_isLeaf) {
            text.append("[").append(m_localModel.dumpLabel(0, m_train)).append("]");
        }
        else {
            this.prefixTree(text);
        }
        
        return text.toString();
    }
    
    /**
     * Returns source code for the tree as an if-then statement. The
     * class is assigned to variable "p", and assumes the tested
     * instance is named "i". The results are returned as two StringBuilders:
     * a section of code for assignment of the class, and a section of
     * code containing support code (eg: other support methods).
     *
     * @param className the classname that this static classifier has
     *
     * @return an array containing two StringBuilders, the first string containing
     * assignment code, and the second containing source for support code.
     */
    public StringBuilder[] toSource(String className) {
        StringBuilder[] result = new StringBuilder[2];
        if (m_isLeaf) {
            result[0] = new StringBuilder("    p = " + m_localModel.distribution().maxClass(0) + ";\n");
            result[1] = new StringBuilder();
        }
        else {
            StringBuilder text    = new StringBuilder();
            StringBuilder atEnd   = new StringBuilder();
            long          printID = nextID();
            text.append("  static double N").append(Integer.toHexString(m_localModel.hashCode())).append(printID)
                    .append("(Object []i) {\n").append("    double p = Double.NaN;\n");
            text.append("    if (").append(m_localModel.sourceExpression(- 1, m_train)).append(") {\n");
            text.append("      p = ").append(m_localModel.distribution().maxClass(0)).append(";\n");
            text.append("    } ");
            
            for (int i = 0; i < m_sons.length; i++) {
                text.append("else if (").append(m_localModel.sourceExpression(i, m_train)).append(") {\n");
                if (m_sons[i].m_isLeaf) {
                    text.append("      p = ").append(m_localModel.distribution().maxClass(i)).append(";\n");
                }
                else {
                    StringBuilder[] sub = m_sons[i].toSource(className);
                    text.append(sub[0]);
                    atEnd.append(sub[1]);
                }
                
                text.append("    } ");
                if (i == m_sons.length - 1) {
                    text.append('\n');
                }
            }
            
            text.append("    return p;\n  }\n");
            result[0] = new StringBuilder("    p = " + className + ".N");
            result[0].append(Integer.toHexString(m_localModel.hashCode())).append(printID).append("(i);\n");
            result[1] = text.append(atEnd);
        }
        
        return result;
    }
    
    /**
     * Returns number of leaves in tree structure.
     *
     * @return the number of leaves
     */
    public int numLeaves() {
        int num = 0;
        if (m_isLeaf) {
            return 1;
        }
        for (ClassifierTree m_son : m_sons) {
            num += m_son.numLeaves();
        }
        return num;
    }
    
    /**
     * Returns number of nodes in tree structure.
     *
     * @return the number of nodes
     */
    public int numNodes() {
        int no = 1;
        if (! m_isLeaf) {
            for (ClassifierTree m_son : m_sons) {
                no += m_son.numNodes();
            }
        }
        return no;
    }
    
    /**
     * Prints tree structure.
     *
     * @return the tree structure
     */
    public String toString() {
        try {
            StringBuilder text = new StringBuilder();
            if (m_isLeaf) {
                text.append(": ");
                text.append(m_localModel.dumpLabel(0, m_train));
            }
            else {
                this.dumpTree(0, text);
            }
    
            text.append("\n\nNumber of Leaves  : \t").append(this.numLeaves()).append("\n");
            text.append("\nSize of the tree : \t").append(this.numNodes()).append("\n");
            return text.toString();
        }
        catch (Exception e) {
            return "Can't print classification tree.";
        }
    }
    
    /**
     * Returns a newly created tree.
     *
     * @param data the training data
     *
     * @return the generated tree
     * @throws Exception if something goes wrong
     */
    protected ClassifierTree getNewTree(Instances data) throws Exception {
        ClassifierTree newTree = new ClassifierTree(m_toSelectModel);
        newTree.buildTree(data, false);
        return newTree;
    }
    
    /**
     * Returns a newly created tree.
     *
     * @param train the training data
     * @param test  the pruning data.
     *
     * @return the generated tree
     * @throws Exception if something goes wrong
     */
    protected ClassifierTree getNewTree(Instances train, Instances test) throws Exception {
        ClassifierTree newTree = new ClassifierTree(m_toSelectModel);
        newTree.buildTree(train, test, false);
        return newTree;
    }
    
    /**
     * Help method for printing tree structure.
     *
     * @param depth the current depth
     * @param text  for outputting the structure
     */
    private void dumpTree(int depth, StringBuilder text) {
        for (int i = 0; i < m_sons.length; i++) {
            text.append("\n");
            
            for (int j = 0; j < depth; j++) {
                text.append("|   ");
            }
            
            text.append(m_localModel.leftSide(m_train));
            text.append(m_localModel.rightSide(i, m_train));
            if (m_sons[i].m_isLeaf) {
                text.append(": ");
                text.append(m_localModel.dumpLabel(i, m_train));
            }
            else {
                m_sons[i].dumpTree(depth + 1, text);
            }
        }
    }
    
    /**
     * Help method for printing tree structure as a graph.
     *
     * @param text for outputting the tree
     */
    private void graphTree(StringBuilder text) {
        for (int i = 0; i < m_sons.length; i++) {
            text.append("N").append(m_id).append("->").append("N").append(m_sons[i].m_id).append(" [label=\"")
                    .append(Utils.backQuoteChars(m_localModel.rightSide(i, m_train).trim())).append("\"]\n");
            if (m_sons[i].m_isLeaf) {
                text.append("N").append(m_sons[i].m_id).append(" [label=\"")
                        .append(Utils.backQuoteChars(m_localModel.dumpLabel(i, m_train))).append("\" ")
                        .append("shape=box style=filled ");
                if (m_train != null && m_train.numInstances() > 0) {
                    text.append("data =\n").append(m_sons[i].m_train).append("\n");
                    text.append(",\n");
                }
                text.append("]\n");
            }
            else {
                text.append("N").append(m_sons[i].m_id).append(" [label=\"")
                        .append(Utils.backQuoteChars(m_sons[i].m_localModel.leftSide(m_train))).append("\" ");
                if (m_train != null && m_train.numInstances() > 0) {
                    text.append("data =\n").append(m_sons[i].m_train).append("\n");
                    text.append(",\n");
                }
                
                text.append("]\n");
                m_sons[i].graphTree(text);
            }
        }
    }
    
    /**
     * Prints the tree in prefix form
     *
     * @param text the buffer to output the prefix form to
     */
    private void prefixTree(StringBuilder text) {
        text.append("[");
        text.append(m_localModel.leftSide(m_train)).append(":");
        
        for (int i = 0; i < m_sons.length; i++) {
            if (i > 0) {
                text.append(",\n");
            }
            text.append(m_localModel.rightSide(i, m_train));
        }
        
        for (int i = 0; i < m_sons.length; i++) {
            if (m_sons[i].m_isLeaf) {
                text.append("[");
                text.append(m_localModel.dumpLabel(i, m_train));
                text.append("]");
            }
            else {
                m_sons[i].prefixTree(text);
            }
        }
        text.append("]");
    }
    
    /**
     * Help method for computing class probabilities of
     * a given instance.
     *
     * @param classIndex the class index
     * @param instance   the instance to compute the probabilities for
     * @param weight     the weight to use
     *
     * @return the laplace probs
     */
    private double getProbsLaplace(int classIndex, Instance instance, double weight) {
        double prob = 0.0D;
        
        if (m_isLeaf) {
            return weight * m_localModel.classProbLaplace(classIndex, instance, - 1);
        }
        
        int treeIndex = m_localModel.whichSubset(instance);
        if (treeIndex == - 1) {
            double[] weights = m_localModel.weights(instance);
            
            for (int i = 0; i < m_sons.length; i++) {
                if (! m_sons[i].m_isEmpty) {
                    prob += m_sons[i].getProbsLaplace(classIndex, instance, weights[i] * weight);
                }
            }
            return prob;
        }
        
        if (m_sons[treeIndex].m_isEmpty) {
            return weight * m_localModel.classProbLaplace(classIndex, instance, treeIndex);
        }
        return m_sons[treeIndex].getProbsLaplace(classIndex, instance, weight);
    }
    
    /**
     * Help method for computing class probabilities of
     * a given instance.
     *
     * @param classIndex the class index
     * @param instance   the instance to compute the probabilities for
     * @param weight     the weight to use
     *
     * @return the probs
     */
    private double getProbs(int classIndex, Instance instance, double weight) {
        double prob = 0.0D;
        if (m_isLeaf) {
            return weight * m_localModel.classProb(classIndex, instance, - 1);
        }
        
        int treeIndex = m_localModel.whichSubset(instance);
        if (treeIndex == - 1) {
            double[] weights = m_localModel.weights(instance);
            
            for (int i = 0; i < m_sons.length; i++) {
                if (! m_sons[i].m_isEmpty) {
                    prob += m_sons[i].getProbs(classIndex, instance, weights[i] * weight);
                }
            }
            return prob;
        }
        
        if (m_sons[treeIndex].m_isEmpty) {
            return weight * m_localModel.classProb(classIndex, instance, treeIndex);
        }
        return m_sons[treeIndex].getProbs(classIndex, instance, weight);
    }
    
    /**
     * Computes a list that indicates node membership
     */
    public double[] getMembershipValues(Instance instance) {
        double[] a = new double[this.numNodes()];
        
        // Initialize queues
        Queue<Double>         queueOfWeights = new LinkedList<>();
        Queue<ClassifierTree> queueOfNodes   = new LinkedList<>();
        queueOfWeights.add(instance.weight());
        queueOfNodes.add(this);
        int index = 0;
    
        // While the queue is not empty
        while (! queueOfNodes.isEmpty()) {
            a[index++] = queueOfWeights.poll();
            ClassifierTree node = queueOfNodes.poll();
            
            assert node != null;
            if (node.m_isLeaf) {
                continue;
            }
            
            // Check for missing value
            int      treeIndex = node.m_localModel.whichSubset(instance);
            double[] weights   = new double[node.m_sons.length];
            if (treeIndex == - 1) {
                weights = node.m_localModel.weights(instance);
            }
            else {
                weights[treeIndex] = 1.0D;
            }
            
            for (int i = 0; i < node.m_sons.length; i++) {
                queueOfNodes.add(node.m_sons[i]);
                queueOfWeights.add(a[index - 1] * weights[i]);
            }
        }
        return a;
    }
    
    /**
     * Returns the revision string.
     */
    public String getRevision() {
        return RevisionUtils.extract("$Revision: 13476 $");
    }
}
