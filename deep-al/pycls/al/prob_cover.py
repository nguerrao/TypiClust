import numpy as np
import pandas as pd
import torch
import pycls.datasets.utils as ds_utils
from sklearn.metrics.pairwise import cosine_similarity
import math


class ProbCover:
    def __init__(self, cfg, lSet, uSet, budgetSize, delta, method, const_threshold, const_threshold_mean, alpha, number_of_samples, number_of_smallest_values_to_consider,  text_embeddings=None):
        self.cfg = cfg
        self.ds_name = self.cfg['DATASET']['NAME']
        self.seed = self.cfg['RNG_SEED']
        self.model_features = self.cfg['MODEL_FEATURES']
        self.normalize = self.cfg['NORMALIZE']
        self.method = method
        self.all_features = ds_utils.load_features(self.ds_name, self.seed, self.model_features)
        self.lSet = lSet
        self.uSet = uSet
        self.budgetSize = budgetSize
        self.delta = delta
        self.text_embeddings=text_embeddings
        self.top_line = self.cfg['TOP_LINE']
        self.const_threshold=const_threshold
        self.const_threshold_mean=const_threshold_mean
        self.alpha=alpha
        self.number_of_samples=number_of_samples
        self.number_of_smallest_values_to_consider = number_of_smallest_values_to_consider
        self.relevant_indices = np.concatenate([self.lSet, self.uSet]).astype(int)
        self.rel_features = self.all_features[self.relevant_indices]

        if self.model_features == "clip":
            self.cfg.counts[0]=self.get_image_counts(self.text_embeddings, self.rel_features)
            self.cfg.threshold[0]= self.distribute_images_evenly(self.cfg.counts[0],6*self.budgetSize) * self.const_threshold 
            self.thresholds=self.compute_thresholds(self.text_embeddings, self.rel_features, len(self.text_embeddings))
        
        self.graph_df = self.construct_graph()

        file_names=[]
        # Reading the filenames.txt file 
        if cfg.DATASET.NAME=="MSCOCO":
            with open('../../scan/results/mscoco/pretext/filenames.txt', 'r') as f:
                lines = f.readlines()

        elif cfg.DATASET.NAME=="PASCALVOC":
            with open('../../scan/results/pascalvoc/pretext/filenames.txt', 'r') as f:
                lines = f.readlines()
        
        for line in lines:
            file_names.append(line.strip())
        
        self.file_names=file_names

        self.rel_file_names = [self.file_names[i] for i in self.relevant_indices]
        

        print("Loading Labels")
        if cfg.DATASET.NAME == "MSCOCO":

            
            top_line_path = cfg.TOPLINE_PATH 
            self.df_labels = pd.read_csv(top_line_path,index_col=0)
            self.class_list=[
                    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
                    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
                    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
                    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
                    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
                ]
    

        elif cfg.DATASET.NAME == "PASCALVOC":
            top_line_path = cfg.TOPLINE_PATH 
            self.df_labels = pd.read_csv(top_line_path,index_col=0)
            self.class_list=['aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']



    def get_cosine_similarity(self,feature_vec_1, feature_vec_2): 
        """
        Calculate and return the cosine similarity between two vectors.
        
        Args:
        feature_vec_1 (numpy.array): First image vector
        feature_vec_2 (numpy.array): Second image vector

        Return:
        float: 1 minus the cosine similarity, ensuring that higher similarity corresponds to a smaller distance and vice versa.
        """
        return (1-cosine_similarity(feature_vec_1, feature_vec_2))
    
    def get_image_counts(self, text_features, image_features):
        """
        Calculate the cosine similarities between text and image features, identify the class of highest similarity for each image, 
        and return the count of images per class.
        
        Args:
        text_features (numpy.array): 2D array where each row represents text feature vector of a class
        image_features (numpy.array): 2D array where each row represents an image feature vector 

        Return:
        counts (numpy.array): Count of images that correspond to each class based on highest similarity
        """
        similarities = cosine_similarity(text_features, image_features)
        class_indices = np.argmax(similarities, axis=0)
        unique_values, counts = np.unique(class_indices, return_counts=True)
        return counts
    
    
    def compute_clip_distance(self, image_features, text_features):
        """
        Compute the cosine similarities between image features and text features.
        
        Args:
        image_features (numpy.array): 2D array where each row represents an image feature vector 
        text_features (numpy.array): 2D array where each row represents text feature vector of a class

        Return:
        numpy.array: 2D array of cosine similarities between each image feature vector and each text feature vector.
        """
        # Calculate the cosine similarities between image and text features
        similarities = cosine_similarity(image_features, text_features)
        return similarities

    def normalize_similarity_matrix(self, matrix):
        """
        Normalize a similarity matrix by subtracting the mean of the three smallest values in each column from each value in that column.
        
        Args:
        matrix (numpy.array): 2D array representing the similarity matrix to be normalized.

        Return:
        numpy.array: 2D array representing the normalized similarity matrix.
        """
        # Transpose the matrix for easier column (image) operations
        matrix = matrix.T

        # Initialize a normalized matrix with the same shape
        normalized_matrix = np.zeros(matrix.shape)
        
        for i in range(matrix.shape[0]):
            # Compute the mean of the three smallest values for each image (column)
            mean_of_mins = np.mean(np.sort(matrix[i])[:3])

            # Normalize each column by subtracting the mean of its three smallest values
            normalized_matrix[i] = matrix[i] - mean_of_mins

        # Return the transposed version of the normalized matrix, so it has the same shape as the input matrix
        return normalized_matrix.T


    def compute_thresholds(self, text_features, image_features, n_classes):
        """
        Compute thresholds for image object classification based on cosine similarity between text and image features.
        Threshold for each class is calculated as the mean of the similarities minus a fraction of the standard deviation.
        
        Args:
        text_features (numpy.array): 2D array where each row represents text feature vector of a class
        image_features (numpy.array): 2D array where each row represents an image feature vector 
        n_classes (int): Number of classes 

        Return:
        thresholds (numpy.array): Array of threshold values for each class 
        """


        # Compute cosine similarities between text and image features
        similarities = cosine_similarity(text_features, image_features)
        
        if self.normalize:
            similarities = self.normalize_similarity_matrix(similarities)

        # Initialize an empty list for each class
        class_similarities = [[] for _ in range(n_classes)]

        # Determine class assignments for each image based on maximum similarity
        classes_indices = np.argmax(similarities, axis=0)

        # Add the similarity score to the corresponding class list
        for i in range(len(image_features)):
            class_index = classes_indices[i]
            class_similarities[class_index].append(similarities[class_index, i])

        # Compute the mean similarity for each class
        means = [np.mean(similarities) for similarities in class_similarities]

        # Compute the standard deviation of similarities for each class
        std_devs = [np.std(similarities) for similarities in class_similarities]

        # Compute thresholds as cte_mean*mean - cte_th*standard deviation
        thresholds_std_dev = [self.const_threshold_mean*mean + self.const_threshold*std_dev for mean, std_dev in zip(means, std_devs)]

        thresholds = np.array(thresholds_std_dev)

        return thresholds

    def extract_values_above_others(self, lst, k, thresh):
        """
        Extract values from a list that are above a certain threshold.
        The threshold is determined based on the highest value in the list and the average of the smallest 'k' values.
        
        Args:
        lst (list): List of numerical values
        k (int): Number of smallest values to consider for average
        thresh (float): Threshold fraction to consider for extraction 

        Return:
        count (int): Count of values in the list that are above the computed threshold.
        """

        lst_sorted = sorted(lst)
        low_val = sum(lst_sorted[:k]) / k
        high_val = max(lst)
        threshold = high_val - (high_val - low_val) * thresh  # Adjust the threshold value as needed
        extracted_values = [value for value in lst if value > threshold]
        count = len(extracted_values)
        return count
    
    def extract_values_above_others_new(self, lst, k, thresh):
        """
        Extract values from a list that are above a certain threshold.
        The threshold is determined based on the highest value in the list and the average of the smallest 'k' values.
        
        Args:
        lst (list): List of numerical values
        k (int): Number of smallest values to consider for average
        thresh (float): Threshold fraction to consider for extraction 

        Return:
        count (int): Count of values in the list that are above the computed threshold.
        """

        lst_sorted = sorted(lst)
        low_val = sum(lst_sorted[:k]) / k
        high_val = max(lst)
        threshold = high_val - (high_val - low_val) * thresh  # Adjust the threshold value as needed
        #extracted_values = [value for value in lst if value > threshold]
        indices = [index for index, value in enumerate(lst) if value > threshold]
        #values = [value for index, value in enumerate(lst) if value > threshold]

        count = [1 if value>threshold else 0 for value in lst]
        
        return count

    def distribute_images_evenly(self, number_of_image_per_class, total_of_selected_images):
        """
        Distribute a total number of selected images evenly among classes, ensuring not to exceed the available images in each class.
        
        Args:
        number_of_image_per_class (numpy.array): Array containing the number of available images for each class
        total_of_selected_images (int): Total number of images to be distributed 

        Return:
        selections (numpy.array): Array containing the number of images assigned to each class.
        """

        num_classes = len(number_of_image_per_class)

        # Initially, try to distribute the images evenly among the classes
        selections = np.full(num_classes, total_of_selected_images// num_classes)

        # If k is not divisible by num_classes, this will distribute the remainder
        selections[:total_of_selected_images % num_classes] += 1

        # This step ensures that we don't select more images than are available in each class
        selections = np.minimum(number_of_image_per_class, selections)

        # This is the number of remaining images to be assigned after the initial assignment
        remaining = total_of_selected_images - np.sum(selections)

        # While we still have images to assign
        while remaining > 0:
            # Find the classes where we haven't selected all images yet
            not_full = number_of_image_per_class > selections

            # If there's no such class, we can't assign any more images
            if not np.any(not_full):
                break

            # Increase the selections for those classes
            selections[not_full] += 1

            # Decrease the remaining images to be assigned
            remaining -= np.count_nonzero(not_full == True)

        return selections

    def get_best_image(self, df_image, counter):
        """
        Select the best image from a dataframe, based on a provided counter dictionary.
        The selection is made by class name, considering classes with lower counts first.
        

        Args:
        df_image (pandas.DataFrame): DataFrame containing images selected with probCover. It should include 'Image Filename' column representing image file names 
                                        and boolean columns for each class name indicating the presence of the class in the image.
        counter (dict): Dictionary with class names as keys and their counts (In the label set) as values.

        Return:
        image_selected (str): The file name of the selected image.
        """
       
        sorted_items = sorted(counter.items(),reverse=True)
        sorted_counter = dict(sorted_items)
        
        for class_name in sorted_counter:
            df_class = df_image[df_image[class_name]>=1]
            
            if not df_class.empty:
                image_selected  = df_class.loc[df_class['count'].idxmax(), 'Image Filename']
                return image_selected


    def construct_graph(self, batch_size=500):
        """
        creates a directed graph where:
        x->y iff l2(x,y) < delta.

        represented by a list of edges (a sparse matrix).
        stored in a dataframe
        """
        xs, ys = [], []
        #xs, ys, ds = [], [], []
        print(f'Start constructing graph using delta={self.delta}')
        # distance computations are done in GPU
        if self.model_features == "clip":
            all_text_similarity = self.compute_clip_distance(self.rel_features, self.text_embeddings)
            cuda_text_feats=torch.from_numpy(all_text_similarity).cuda() 
        
        cuda_feats = torch.from_numpy(self.rel_features).cuda() 
        for i in range(len(self.rel_features) // batch_size):
            # distance comparisons are done in batches to reduce memory consumption
            cur_feats = cuda_feats[i * batch_size: (i + 1) * batch_size]
            dist_image = torch.cdist(cur_feats, cuda_feats)

            if self.model_features == "clip":
            # distance between similarities
                cur_text_feats = cuda_text_feats[i * batch_size: (i + 1) * batch_size]
                dist_text= torch.cdist(cur_text_feats, cuda_text_feats)
                #dist=torch.Tensor(self.get_cosine_similarity(cur_feats.cpu(), cuda_feats.cpu())).cuda()
                dist=self.alpha*dist_image + (1-self.alpha)*dist_text
            
            else:
                #dist=torch.Tensor(self.get_cosine_similarity(cur_feats.cpu(), cuda_feats.cpu())).cuda()
                dist=dist_image 

            mask = dist < self.delta
            # saving edges using indices list - saves memory.
            x, y = mask.nonzero().T
            xs.append(x.cpu() + batch_size * i)
            ys.append(y.cpu())
            #ds.append(dist[mask].cpu())

        xs = torch.cat(xs).numpy()
        ys = torch.cat(ys).numpy()
        #ds = torch.cat(ds).numpy()
        

        df = pd.DataFrame({'x': xs, 'y': ys})
        #df = pd.DataFrame({'x': xs, 'y': ys, 'd': ds})
        print(f'Finished constructing graph using delta={self.delta}')
        print(f'Graph contains {len(df)} edges.')
        return df

    def select_samples(self):
        """
        selecting samples using the greedy algorithm.
        iteratively:
        - removes incoming edges to all covered samples
        - selects the sample high the highest out degree (covers most new samples)

        """
        print(f'Start selecting {self.budgetSize} samples.')
        selected = []
        # removing incoming edges to all covered samples from the existing labeled set
        edge_from_seen = np.isin(self.graph_df.x, np.arange(len(self.lSet)))
        covered_samples = self.graph_df.y[edge_from_seen].unique()
        cur_df = self.graph_df[(~np.isin(self.graph_df.y, covered_samples))]
        for i in range(self.budgetSize):
            self.cfg.coverage = len(covered_samples) / len(self.relevant_indices)
            # selecting the sample with the highest degree
            degrees = np.bincount(cur_df.x, minlength=len(self.relevant_indices))
            print(f'Iteration is {i}.\tGraph has {len(cur_df)} edges.\tMax degree is {degrees.max()}.\tCoverage is {self.cfg.coverage:.3f}')

            cur = degrees.argmax() 
            
            if degrees.max() <= 1:
                # Get the indices of elements above 0
                indices_degrees = [index for index, value in enumerate(degrees) if value > 0]

                # Randomly select an index from the list of indices
                cur = np.random.choice(indices_degrees)
                
            # Count objects
            if 'clip_selection_max_object' in self.method:
                
                indices_images=degrees.argsort()[::-1][:self.number_of_samples]

                if self.top_line:

                    image_names=[self.rel_file_names[image] for image in indices_images]
                
                    df_image_selection=self.df_labels[self.df_labels['Image Filename'].isin(image_names)].copy()

                    if self.cfg.TOPLINE_COUNT_METHOD == 'per_class':
                        image_selected=df_image_selection['Image Filename'][df_image_selection['count_diff_class'].idxmax()]

                    if self.cfg.TOPLINE_COUNT_METHOD == 'per_object':
                        image_selected=df_image_selection['Image Filename'][df_image_selection['count'].idxmax()]
                    
                    if self.cfg.TOPLINE_COUNT_METHOD == 'per_object_weighted':
            
                        # add weight to each class
                        # Find the maximum value in the dictionary
                        max_value = max(self.cfg.count_class[0].values())

                        # Create a new dictionary with weights for each class, (count if count != 0 else 10)
                        # but first check if the count is zero and set it to a default value if needed
                        weights = {cls:  (max_value/count if count != 0 else 10) for cls, count in self.cfg.count_class[0].items()}
                        
                        df_image_selection['count_weighted'] = df_image_selection.apply(lambda row: sum(row[col] * weights[col] for col in self.class_list), axis=1)
                 
                        image_selected=df_image_selection['Image Filename'][df_image_selection['count_weighted'].idxmax()]    
                        # Update counter:
                        info_image = df_image_selection[df_image_selection['Image Filename']==image_selected]
                        for class_name in self.class_list:
                            self.cfg.count_class[0][class_name] += info_image[class_name].values[0]
                    cur = indices_images[image_names.index(image_selected)]

                    # Add a weight to each class, weight proportional to best class
                    # Do that but use count instead of diff_class 
                    
                        # -> Check performance
                        
                else:
     
                    similarities=cosine_similarity(self.text_embeddings,self.rel_features[indices_images])
                    
                    if 'v2' in self.method:

                        num_images = similarities.shape[1]  # Get the number of images (N in this case)
                        class_counts = []

                        #compute weights:
                        max_value = max(self.cfg.count_class[0].values())
                        #weights = {cls:  (max_value/count if count != 0 else 10) for cls, count in self.cfg.count_class[0].items()}
                        weights = {cls:  1 for cls, count in self.cfg.count_class[0].items()}
                        
                        score_max=0
                        for image_index in range(num_images):
                            image_similarities = similarities[:, image_index]  # Get the similarity values for the current image    
                            #class_counts.append(self.extract_values_above_others(image_similarities, self.number_of_smallest_values_to_consider, self.const_threshold))
                            count_list = self.extract_values_above_others_new(image_similarities, self.number_of_smallest_values_to_consider, self.const_threshold)
                           
                            score = 0
                            for i in range(len(self.class_list)):
                                score += count_list[i]*weights[self.class_list[i]]
                            if score>score_max:
                                count_list_final = count_list
                                score_max = score

                            class_counts.append(score)
                        for i, count_class in enumerate(count_list_final):
                            self.cfg.count_class[0][self.class_list[i]] += count_class
                            
                    else:
                        print('================= USING MAX OBJECT ORIGINAL =================')
                        if self.normalize:
                            similarities = self.normalize_similarity_matrix(similarities)

                        class_counts = np.sum(similarities > self.thresholds[:, None], axis=0) 


                    # Get indices of images sorted by class count in descending order
                    sorted_image_indices = np.argsort(class_counts)[::-1]
            
                    cur = indices_images[sorted_image_indices[0]]
                    #print("cur", cur)
     
            
        
            elif self.method == 'clip_selection_balanced_classes':

                if self.top_line:

                    # Get best image 
                    indices_images=degrees.argsort()[::-1][:self.number_of_samples]
                    
                    image_names=[self.rel_file_names[image] for image in indices_images]

                    df_image_selection=self.df_labels[self.df_labels['Image Filename'].isin(image_names)]

                    image_selected = self.get_best_image(df_image_selection, self.cfg.count_class[0])
                    
                    info_image = df_image_selection[df_image_selection['Image Filename']==image_selected]
           
                
                    # Update counter:
                    for class_name in self.class_list:
                        self.cfg.count_class[0][class_name] += info_image[class_name].values[0]
            
                
                    cur = indices_images[image_names.index(image_selected)]
                    

                else:

                    check=True
                    while(check):
                        
                        similarities=cosine_similarity(self.text_embeddings,self.rel_features[cur].reshape(1, -1))
                        class_idx=np.argmax(similarities, axis=0)
                        class_name = self.class_list[int(class_idx[0])]

                        if self.cfg.count_class[0][class_name]<=self.cfg.threshold[0][int(class_idx[0])]:
                            print("================= SAMPLE SELECTED =====================")
                            self.cfg.count_class[0][class_name]+=1
                            check=False
                        else:
                            degrees[cur]=0
                            cur=degrees.argmax() 

                
            # cur = np.random.choice(degrees.argsort()[::-1][:5]) # the paper randomizes selection 

         
            # removing incoming edges to newly covered samples
            new_covered_samples = cur_df.y[(cur_df.x == cur)].values
            assert len(np.intersect1d(covered_samples, new_covered_samples)) == 0, 'all samples should be new'
            cur_df = cur_df[(~np.isin(cur_df.y, new_covered_samples))]
            covered_samples = np.concatenate([covered_samples, new_covered_samples])
            selected.append(cur)
        assert len(selected) == self.budgetSize, 'added a different number of samples'
        activeSet = self.relevant_indices[selected]
        remainSet = np.array(sorted(list(set(self.uSet) - set(activeSet))))
        print("remainSet is", remainSet)
        print(f'Finished the selection of {len(activeSet)} samples.')
        print(f'Active set is {activeSet}')
        print(f"Class counts using {self.method} method", self.cfg.count_class[0])
   
        return activeSet, remainSet
