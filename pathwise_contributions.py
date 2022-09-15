import itertools
import re
import numpy as np
import pandas as pd
from lime_explainer import LimeTabularExplainerReview as LimeTabularExplainer,pyplot_figure
# Pathwise-contribution Algorithm
def PathwiseContributionLimeExplanations(sample,pred_paths,
                                     pipeline_utils,
                                     is_classification=True,
                                     collapse_path_splits=False,
                                     epsilon=1e-10,):
    records={}
    
    # Get all the information such as the models and paths from the pred_paths
    p_paths = re.findall('\w\d+',' '.join(list(pred_paths.keys())))
    models =[list(pred_paths[p]['sub_models'].keys()) for p in p_paths]
    paths_features = {p:[f['features'] for m,f in pred_paths[p]['sub_models'].items()] for p in p_paths}
    
    paths_features = {p:list(itertools.chain(*v)) for p,v in paths_features.items()}
    #print(paths_features)
    #for p,v in paths_features.items():
    #    
    nb_models = len(models)+1 # +1 for the final model
    
    nb_paths = len(p_paths)
    
    features = pipeline_utils['pipelined_features']
    feature_explanation_table = pd.DataFrame(np.zeros((len(features)+nb_models+1,
                                                       nb_paths+1)), 
                                                      columns= [f'P{i+1}' for i in range(nb_paths)]+['MF'])
    
    feature_explanation_table.index = features+[f'M{i+1}' for i in range(nb_models)]+['Cv']
    
    
    # Compute the blind-black-box explanations where the explainer only sees the inputs and output of the pipeline
    blind_level_analyzer = pipeline_utils['blind_pipeline']
    pred= blind_level_analyzer['pred_fn'](sample).argmax(-1)[0]
    blind_explainer = blind_level_analyzer['explainer']
    blind_level_explanation= blind_explainer.explain_instance(sample[0],
                                                              predict_fn=blind_level_analyzer['pred_fn'],
                                                              labels=[pred])
    
    
    #records['blind_black-box_intercepts']= blind_level_explanation.intercept[pred]
    #records['sampled_data'] = blind_level_explanation.sampled_data
    records['blind_explainer'] = blind_level_explanation
    mf_attributions = dict(list( blind_level_explanation.local_exp.values())[0])
    for idx, f in enumerate(features):
        att = mf_attributions[idx]
        feature_explanation_table.loc[f,['MF']]  =  att
        for p in p_paths:
            if f in paths_features[p]:
                feature_explanation_table.loc['Cv',p]+= att
    
    
    # Compose the feature Dataframe for the sample
    sample_df = pd.DataFrame(sample,columns=features)
    
    sub_models_prediction =  {}
    paths_elements = {}
    # Collect the attributions across each path
    for path_name,elements in pred_paths.items():
        sub_models = elements['sub_models']
        previous_model ={}
        paths_elements[path_name]=[]
        for model_name,models_def in sub_models.items():
            input_features = models_def['features']
            model_explainer = models_def['explainer']
            model_pred_fn = models_def['pred_fn']
            prev_models = set(previous_model.keys())
            
            paths_elements[path_name] += input_features+[model_name]
            
            # Check if the model's input features is not from a previous model on the pipeline
            commons = set([s.lower() for s in sub_models_prediction.keys()]).intersection([d.lower() for d in input_features])
            raw_features = [f for f in input_features if f in features]
            intermediate_output_features =  [f for f in input_features if f not in features]
            if len(commons)<1:
                # Then there is no previous model that this model is connected to
                data_sample = sample_df[input_features].values
            else:
                # If the model is dependent on a previous model
                data_sample = sample_df[raw_features]
                previous_model_preds = pd.DataFrame(np.asarray([[p[0] for k,p in sub_models_prediction.items() if k in intermediate_output_features]]),columns=intermediate_output_features)
                data_sample = pd.concat([data_sample,previous_model_preds],axis=1)[input_features].values
            
            # Get the prediction from the model
            pred_label = model_pred_fn(data_sample).argmax(-1)[0] if is_classification else model_pred_fn(data_sample)   
            previous_model[model_name] = pred_label
            sub_models_prediction[model_name] = [pred_label]
            
            # Explain the prediction decision
            instance_explanation = model_explainer.explain_instance(data_sample[0],
                                                                    model_pred_fn,
                                                                    labels=[pred_label])
            records[f'intercept_{model_name}']=instance_explanation.intercept[pred_label]
            #instance_explanation.show_in_notebook()
            #plt.pause(0.0002)
            m1_attributions = dict(list(instance_explanation.local_exp.values())[0])
            for idx, f in enumerate(input_features):
                att = m1_attributions[idx]
                #m1_attributions_array[0,idx] = att
                feature_explanation_table.loc[f,[path_name]]  =  att
                
    
    # Pass everything through the decision network
    decision_network = pipeline_utils['decision_model']
    input_features = decision_network['features']
    model_explainer = decision_network['explainer']
    model_pred_fn = decision_network['pred_fn']
    
    
    # Get the features
    raw_features = [f for f in input_features if f in features]
    
    if len(raw_features)>0:
        # Get the features from the sample_df 
        data_sample = sample_df[raw_features]
        subModel_outputs = pd.DataFrame(sub_models_prediction)
        data_sample = pd.concat([data_sample,subModel_outputs],axis=1)[input_features].values
    else:
        data_sample = pd.DataFrame(sub_models_prediction)#[f.upper() for f in input_features]
        data_sample = data_sample[[f.upper() for f in input_features]].values
    
    # Get the prediction from the model
    pred_label = model_pred_fn(data_sample).argmax(-1)[0] if is_classification else model_pred_fn(data_sample) 
    sub_models_prediction['decision_model'] = [pred_label]
        
    # Explain the prediction decision
    instance_explanation = model_explainer.explain_instance(data_sample[0],
                                                                    model_pred_fn,
                                                                    labels=[pred_label])
    records[f'intercept_dn']=instance_explanation.intercept[pred_label]
    my_attributions = dict(list(instance_explanation.local_exp.values())[0])
    for idx, f in enumerate(input_features):
        att = my_attributions[idx]
        for p in p_paths:
            if f in paths_elements[p] or f.upper() in paths_elements[p]:
                if f in raw_features:
                    feature_explanation_table.loc[f,p] = att
                else:
                    feature_explanation_table.loc[f.upper(),p] = att
        
    
    # Given the feature explanation table along the prediction paths, estimate the feature specific explanation w.r.t the entire
    # prediction pipeline
    
    # Compute the path explanation factor
    fa=(feature_explanation_table.iloc[-1,:]/(feature_explanation_table.iloc[:-1,:].sum()+epsilon)).values
    #print(fa)
    records['attribution_factor'] = fa
    
    # Compute the final attribution with respect to each element across all the paths
    attributions =  np.dot(feature_explanation_table.iloc[:-1,:].values,fa)
    
    # Make the cummulative condition is obeyed
    #print(round(attributions.sum(),3),)
    assert round(attributions.sum(),3) == round(feature_explanation_table.iloc[-1,:].sum(),3), "Invalid computation"
    attributions = pd.DataFrame(attributions.reshape(-1,1),columns=['Attributions'],index=feature_explanation_table.index.to_list()[:-1])
    
    if collapse_path_splits:
        feature_split = pipeline_utils['feature_splits']
        for f,k in feature_split.items():
            attributions.loc[f,'Attributions'] = attributions.loc[k,'Attributions'].sum()
            attributions.drop(index=k,inplace=True)
    
    
    
    
    
    return  feature_explanation_table,attributions,sub_models_prediction,records



#### Collapse all the reminants into their paths by sharing their contributions as the joint contribution across all the elements on the path
def distributeLocalContributions(pipeline_utils,
                                 label,
                                 attributions,
                                     predictors,
                                     paths_information,
                                     is_shapely=False,
                                     verbose=True,
                                collapse_path_splits=False,
                                ):
    zzb=attributions.copy(deep=True)
    index_to_drop=['base_v'] if is_shapely else []
    for p,data in paths_information.items():
        
        
        for m,feats in data['sub_models'].items():
            f= (attributions.loc[feats]**2).sum().values + 1e-10
            index_to_drop.append(m)
            zzb.loc[feats]=(((attributions.loc[feats]**2)/f)*attributions.loc[m] + attributions.loc[feats])
    
    if is_shapely:
        # Share the 'base_v' score between the entire paths across all the predictors
        fg=(zzb.loc[predictors]**2).sum().values + 1e-10
        zzb.loc[predictors]=(((zzb.loc[predictors]**2)/fg)*zzb.loc['base_v'] + zzb.loc[predictors])
    
    zzb.drop(index=index_to_drop,inplace=True)
    
    if collapse_path_splits:
        feature_split = pipeline_utils['feature_splits']
        for f,k in feature_split.items():
            zzb.loc[f,'Attributions'] = zzb.loc[k,'Attributions'].sum()
            zzb.drop(index=k,inplace=True)
    
    if verbose:
        att=list(zzb.loc[predictors].to_records())
        att= sorted(att,key=lambda x:abs(x[1]))
        pyplot_figure(att,f'Y = {label}');
    
    return zzb

def collapseAttributions(feature_splits,attributions):
    for f,k in feature_splits.items():
            attributions.loc[f,'Attributions'] = attributions.loc[k,'Attributions'].sum()
            attributions.drop(index=k,inplace=True)
    return attributions