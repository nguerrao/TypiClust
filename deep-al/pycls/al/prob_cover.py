import numpy as np
import pandas as pd
import torch
import pycls.datasets.utils as ds_utils
from sklearn.metrics.pairwise import cosine_similarity
import math


class ProbCover:
    def __init__(self, cfg, lSet, uSet, budgetSize, delta, clip_selection, const_threshold, text_embeddings=None):
        self.cfg = cfg
        self.ds_name = self.cfg['DATASET']['NAME']
        self.seed = self.cfg['RNG_SEED']
        self.all_features = ds_utils.load_features(self.ds_name, self.seed)
        print("self.all_features ", len(self.all_features ))
        self.lSet = lSet
        self.uSet = uSet
        self.budgetSize = budgetSize
        self.delta = delta
        self.text_embeddings=text_embeddings
        #self.count_class={i: 0 for i in range(self.num_class)}
        self.clip_selection=clip_selection
        self.const_threshold=const_threshold
        #self.thresold=math.ceil((6*budgetSize)//self.cfg.num_class) * 1.2 #boucle infini
        
        self.relevant_indices = np.concatenate([self.lSet, self.uSet]).astype(int)
        print("self.relevant_indices  ", len(self.relevant_indices))
        self.rel_features = self.all_features[self.relevant_indices]
        self.cfg.counts[0]=self.get_image_counts(self.text_embeddings, self.rel_features)
        print("self.counts", self.cfg.counts[0])
        self.cfg.threshold[0]= self.distribute_images_evenly(self.cfg.counts[0],6*self.budgetSize) * self.const_threshold
        print("self.threshold", self.cfg.threshold[0])
        self.graph_df = self.construct_graph()


    def get_cosine_similarity(self,feature_vec_1, feature_vec_2): 
        return (1-cosine_similarity(feature_vec_1, feature_vec_2))
    
    def get_image_counts(self, text_features, image_features):
        similarities = cosine_similarity(text_features, image_features)
        class_indices = np.argmax(similarities, axis=0)
        unique_values, counts = np.unique(class_indices, return_counts=True)
        return counts
    
    

    def distribute_images_evenly(self, number_of_image_per_class, total_of_selected_images):
        # Number of classes
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
        cuda_feats = torch.from_numpy(self.rel_features).cuda() #torch.tensor(self.rel_features).cuda() #torch.from_numpy(self.rel_features).cuda()
        for i in range(len(self.rel_features) // batch_size):
            # distance comparisons are done in batches to reduce memory consumption
            cur_feats = cuda_feats[i * batch_size: (i + 1) * batch_size]
            dist = torch.cdist(cur_feats, cuda_feats)
            #dist=torch.Tensor(self.get_cosine_similarity(cur_feats.cpu(), cuda_feats.cpu())).cuda()
            #print("dist is", dist)
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
            # print("degrees",len(degrees ))
            # print("degrees",len(degrees ))
            # print("len cur_dr", len(cur_df))
            cur = degrees.argmax() 
            #cur = np.random.choice(degrees.argsort()[::-1][:5])
            

            if degrees.max() == 1:
                #Get the indices of elements above 0
                indices_degrees = [index for index, value in enumerate(degrees) if value > 0]

                #Randomly select an index from the list of indices
                cur = np.random.choice(indices_degrees)
                

        
            if self.clip_selection==True:
                check=True
                while(check):
                    
                    similarities=cosine_similarity(self.text_embeddings,self.rel_features[cur].reshape(1, -1))
                    class_idx=np.argmax(similarities, axis=0)
                    #print("class_idx", class_idx)
                    #print("elf.count_class[class_idx]",self.count_class[int(class_idx[0])])
                    #print("self.cfg.count_class[0]", self.cfg.count_class[0])

                    if self.cfg.count_class[0][int(class_idx[0])]<=self.cfg.threshold[0][int(class_idx[0])]:
                        print("================= SAMPLE SELECTED =====================")
                        #print("class selected", int(class_idx[0]))
                        self.cfg.count_class[0][int(class_idx[0])]+=1
                        check=False
                    else:
                        #print("================= SAMPLE NOT SELECTED =====================")
                        #print("class not selected", int(class_idx[0]))
                        #degrees.remove(cur)
                        degrees[cur]=0
                        #print("degrees[cur]", degrees[cur])
                        #degrees[degrees != cur]
                        cur=degrees.argmax() 

                # if count[idx]<=TH:
                #     coun[idx]+=1
                #     check=False
                # else:
                #     degrees.remove(cur)
                #     cur=degrees.argmax() 
            
            #print("cur", cur)

                #print("lendeg", len(degrees))
                
            # cur = np.random.choice(degrees.argsort()[::-1][:5]) # the paper randomizes selection 

            # removing incoming edges to newly covered samples
            new_covered_samples = cur_df.y[(cur_df.x == cur)].values
            assert len(np.intersect1d(covered_samples, new_covered_samples)) == 0, 'all samples should be new'
            cur_df = cur_df[(~np.isin(cur_df.y, new_covered_samples))]
            print("len cur_dr", len(cur_df))

            covered_samples = np.concatenate([covered_samples, new_covered_samples])
            selected.append(cur)
        #print("selected", selected)
        assert len(selected) == self.budgetSize, 'added a different number of samples'
        #print("selected", selected)
        activeSet = self.relevant_indices[selected]
        #print("activeSet", activeSet)
        #print("aself.relevant_indicet", self.relevant_indices)
        remainSet = np.array(sorted(list(set(self.uSet) - set(activeSet))))
        print("remainSet is", remainSet)
        print(f'Finished the selection of {len(activeSet)} samples.')
        print(f'Active set is {activeSet}')
        print("class_Count", self.cfg.count_class[0])
        


        return activeSet, remainSet
