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

import weka.core.Instances;
import weka.core.RevisionHandler;

import java.io.Serializable;

/**
 * Abstract class for model selection criteria.
 *
 * @author Eibe Frank (eibe@cs.waikato.ac.nz)
 * @version $Revision: 8034 $
 */
public abstract class ModelSelection implements Serializable, RevisionHandler {
    /** for serialization */
    private static final long serialVersionUID = - 8301330936117736456L;
    
    /**
     * Default Constructor
     */
    public ModelSelection() {}
    
    /**
     * Selects a model for the given dataset.
     *
     * @param train Training data
     *
     * @return a split model for the given data
     * @throws Exception if model can't be selected
     */
    public abstract ClassifierSplitModel selectModel(Instances train) throws Exception;
    
    /**
     * Selects a model for the given train data using the given test data
     *
     * @param train training data
     * @param test  testing data
     *
     * @return a split model for the given data
     * @throws Exception if model can't be selected
     */
    public ClassifierSplitModel selectModel(Instances train, Instances test) throws Exception {
        throw new Exception("Model selection method not implemented");
    }
}
