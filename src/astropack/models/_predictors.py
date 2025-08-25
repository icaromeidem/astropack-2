import numpy as np
import pandas as pd

from astropack.preprocess import (calculate_abs_mag,
                                  assemble_work_df)

class Predictor:
    def __init__(self, id_col, mag_cols, err_cols, dist_col, correction_pairs, models, mc_reps):
        self.id_col = id_col
        self.mag_cols = mag_cols
        self.err_cols = err_cols
        self.dist_col = dist_col
        self.correction_pairs = correction_pairs
        self.models = models
        self.mc_reps = mc_reps

    def predict_parameters(self, args):
        input_data, output_path, keep_cols, save_mode, header = args

        if type(input_data) == str:
            input_data = pd.read_csv(input_data)

        if self.dist_col:
            input_data = calculate_abs_mag(input_data, self.mag_cols, self.dist_col)

        input_data.set_index(self.id_col, drop = True, inplace = True)

        work_df = assemble_work_df(df = input_data,
                                   filters = self.mag_cols,
                                   correction_pairs = self.correction_pairs,
                                   add_colors = True,
                                   verbose = False)


        final_df = pd.DataFrame(index = input_data.index,
                                columns = list(self.models.keys()) + \
                                          [f'{x}-ERR' for x in self.models.keys()])

        for model_name in self.models:
            pipeline = self.models[model_name]
            
            if 'Classifier' in str(type(pipeline[-1])):
                true_predictions = pipeline.predict_proba(work_df)
                true_predictions = [x[1] for x in true_predictions]
            elif 'Regressor' in str(type(pipeline[-1])):
                true_predictions = pipeline.predict(work_df)

            mc_predictions = np.empty(shape = (self.mc_reps, len(work_df)))
            for mc_index in range(0, self.mc_reps):
                norm_dist = np.random.normal(size = (len(input_data), len(self.err_cols)))

                mc_input_data = input_data[self.mag_cols] + \
                                (input_data[self.err_cols] * norm_dist).values
                
                if self.correction_pairs:
                    correction_cols = [x for x in self.correction_pairs.values()]
                    mc_input_data[correction_cols] = input_data[correction_cols]

                mc_work_df = assemble_work_df(df = mc_input_data,
                                              filters = self.mag_cols,
                                              correction_pairs = self.correction_pairs,
                                              add_colors = True,
                                              verbose = False)

                if 'Classifier' in str(type(pipeline[-1])):
                    mc_predictions[mc_index] = [x[1] for x in pipeline.predict_proba(mc_work_df)]
                elif 'Regressor' in str(type(pipeline[-1])):
                    mc_predictions[mc_index] = pipeline.predict(mc_work_df)

            final_df[model_name] = true_predictions
            final_df[f'{model_name}-ERR'] = mc_predictions.std(axis = 0)

        final_df[['RA', 'DEC']] = input_data[['RA', 'DEC']]
        final_df[keep_cols] = input_data[keep_cols]
        final_df = final_df[['RA', 'DEC'] + \
                            [f'{x}{y}' for x in self.models.keys() for y in ['', '-ERR']] + \
                            keep_cols]

        final_df.to_csv(output_path,
                        mode = save_mode,
                        header = header)

        return True