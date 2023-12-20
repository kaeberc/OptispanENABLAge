import pickle
import numpy as np
import shap
from SHAP_age_exponential import SHAP_Age

SEX_NAME = "?"
AGE_FEATURE_NAME = "?"
MODEL_PATH = "?"


class optiSHAP:
    def __init__(self, input_data, back_data):
        self._input_data = input_data
        self._back_data = back_data
        self._model = pickle.load(open(MODEL_PATH, "rb"))
        
        # Make data compatible with model
        optiSHAP._preprocData(input_data)
        
        # Filter data by sex
        self._filterSex()
        
        # Preprocess data
        self._ages = (None, None)
        self._results = (None, None)
        self._preprocSex(0)
        self._preprocSex(1)
        
    
    def _preprocData(data):
        raise Exception("Not Implemented")
    
    def _filterSex(self):
        back_data = self._back_data
        fore_data = self._input_data
        
        back_data_female = back_data[back_data[SEX_NAME]==0]
        back_data_male = back_data[back_data[SEX_NAME]==1]
        fore_data_female = fore_data[fore_data[SEX_NAME]==0]
        fore_data_male = fore_data[fore_data[SEX_NAME]==1]
        
        self._fore = (fore_data_male, fore_data_female)
        self._back = (back_data_male, back_data_female)
        
    def _preprocSex(self, sex = None):
        self.results[None]
        
        fore_data_ori = self._fore[sex]
        back_data_ori = self._back[sex]
        
        fore_age_round = fore_data_ori[AGE_FEATURE_NAME].apply(lambda x: np.round(x))
        back_age_round = back_data_ori[AGE_FEATURE_NAME].apply(lambda x: np.round(x))
        
        
        age_list = sorted(list(set(back_age_round)))
        fore_age_list = sorted(list(set(fore_age_round)))
        shap_values_all_dict = {}
        expected_value = {}
        for age in age_list:
            print(age)
            fore_data_temp = fore_data_ori[fore_age_round==age]
            back_data_temp = back_data_ori[back_age_round==age]
            
            pre = self._model.predict(back_data_temp, output_margin=True)
            expected_value[age] = np.median(pre)  # before used median
            
            explainer = shap.TreeExplainer(self._model, data=back_data_temp)
            shap_values_all_dict[age] = explainer.shap_values(fore_data_temp, per_reference=True)
                    
        back_prediction = self._model.predict(back_data_ori, output_margin=True)
        fore_prediction = self._model.predict(fore_data_ori, output_margin=True)
        model_predict_min = min(back_prediction.min(), fore_prediction.min())
        model_predict_max = max(back_prediction.max(), fore_prediction.max())
        expected_value_list = [expected_value[age] for age in age_list]
        
        # Calculate SHAP age
        shap_age = SHAP_Age()
        shap_age.fit(back_prediction,model_predict_min, expected_value_list, age_list)
        
        # Store SHAP Age
        self._ages[sex] = shap_age.get_shap_age(fore_prediction)

        # Calculate relative attribution
        fore_final_attr_dict, fore_shap_age_dict, back_shap_age_dict = shap_age.convert_shap_values(model_train, shap_values_all_dict, fore_prediction, back_prediction, fore_age_round, back_age_round)
        self._results[sex] = fore_final_attr_dict, back_shap_age_dict, fore_data_ori, fore_age_round
        

    def getAge(self):
        return self._ages
    
    def individualPlot(self, individual, age, sex, save_path):
        fore_final_attr_dict, back_shap_age_dict, fore_data, fore_age_round= self._results[sex] 
        idx = individual
        SHAP_Age().individualized_plot(back_shap_age_dict[age].mean(), fore_final_attr_dict[age][idx,:], fore_data[fore_age_round==age].iloc[idx,:], display_col, show=False, save_path=save_path+'/individualized_plot_age_'+str(age)+'_index_'+str(idx)+'.pdf')

    
if __name__ == "__main__":
    s = optiSHAP(inpt, background)
    fem_ages = s.calcAge(1)
    masc_ages = s.calcAge(0)