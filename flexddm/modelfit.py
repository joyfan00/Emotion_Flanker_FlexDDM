from .models.Model import Model
import matplotlib.pyplot as plt
import sys
import pandas as pd
from tqdm.notebook import tqdm
from ._utilities import convertToDF, getRTData
import seaborn as sns
import os


def fit(models, input_data, startingParticipants=None, endingParticipants=None,
        input_data_id="PPT", input_data_condition="Condition", input_data_rt="RT",
        input_data_accuracy="Correct", output_fileName='output.csv', return_dataframes=False,
        posterior_predictive_check=True):

    output_dir = "fit"
    os.makedirs(output_dir, exist_ok=True)
    fit_parameters_dir = os.path.join(output_dir, "fitted_parameters")
    os.makedirs(fit_parameters_dir, exist_ok=True)

    if posterior_predictive_check:
        posterior_predictive_check_dir = os.path.join(output_dir, "posterior_predictive_check")
        os.makedirs(posterior_predictive_check_dir, exist_ok=True)

    # Load input data if a path is given
    if isinstance(input_data, str):
        input_data = getRTData(path=input_data, input_data_id=input_data_id, input_data_condition=input_data_condition,
                               input_data_rt=input_data_rt, input_data_accuracy=input_data_accuracy)

    if startingParticipants is None and endingParticipants is None:
        startingParticipants = input_data['id'].min()
        endingParticipants = input_data['id'].max()

    dflist = []

    for model in models:
        df = pd.DataFrame(columns=['id'] + model.parameter_names + ['X^2', 'bic'])

        pbar = tqdm(range(startingParticipants, endingParticipants + 1)) if endingParticipants - startingParticipants > 1 else range(startingParticipants, endingParticipants + 1)
        pbar.set_description("Fitting Model to Data")

        for id in pbar:
            if input_data[input_data['id'] == id].empty:
                continue

            # # Fit parameters
            fitstat = sys.maxsize - 1
            fitstat2 = sys.maxsize
            pars = None
            runint = 1

            while fitstat != fitstat2:
                fitstat2 = fitstat
                pars, fitstat = model.fit(model.modelsimulationfunction, input_data[input_data['id'] == id], pars, run=runint)
                runint += 1
            # pars = [0.35334771, 0.74951479, 0.20180556, 3.43341415, 0.730947, 0.30971389, -0.28920715, 0.20907516, 0.16146786]
            print(pars)

            # # Get quantiles dynamically from the model
            quantiles_cdf = model.QUANTILES_CDF if hasattr(model, "QUANTILES_CDF") else [.10, .30, .50, .70, .90]
            quantiles_caf = model.QUANTILES_CAF if hasattr(model, "QUANTILES_CAF") else [.25, .50, .75]

            current_input = input_data[input_data['id'] == id]
            myprops = model.proportions(current_input, quantiles_cdf, quantiles_caf)

            # # Compute BIC using the refactored model function
            # def model_function(x, props, param_number, parameter_names, function, data, bounds, final=False):
            bic = Model.model_function(pars, myprops, model.param_number, model.parameter_names,
                           model.modelsimulationfunction, current_input, model.bounds, final=True)
            print(f'Participant {id}, BIC = {bic}')

            df.loc[len(df)] = [id] + list(pars) + [fitstat, bic]

            # if posterior_predictive_check:
            #     posterior_predictive_check_model_dir = os.path.join(posterior_predictive_check_dir, model.__class__.__name__)
            #     os.makedirs(posterior_predictive_check_model_dir, exist_ok=True)

            #     # Run model simulation with fitted parameters
            #     res = model.modelsimulationfunction(*pars, nTrials=len(current_input))
            #     simulated_rts = convertToDF(res, id)

            #     # Prepare combined DataFrame
            #     rt_data = pd.DataFrame({
            #         'experimental_rts': current_input["rt"].tolist(),
            #         'experimental_congruency': current_input['condition'].tolist(),
            #         'experimental_accuracy': current_input['accuracy'].tolist(),
            #         'simulated_rts': simulated_rts['rt'].tolist(),
            #         'simulated_congruency': simulated_rts['condition'].tolist(),
            #         'simulated_accuracy': simulated_rts['accuracy'].tolist()
            #     })

            #     fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
            #     labels_added = set()

            #     unique_conditions = current_input[input_data_condition].unique()

            #     conditions = [(cond, acc, ax) for (cond, ax_row) in zip(unique_conditions, axes) for (acc, ax) in zip([1, 0], ax_row)]

            #     for (congruency, accuracy, ax) in conditions:
            #         experimental_rt_data = rt_data[(rt_data['experimental_congruency'] == congruency) & (rt_data['experimental_accuracy'] == accuracy)]
            #         simulated_rt_data = rt_data[(rt_data['simulated_congruency'] == congruency) & (rt_data['simulated_accuracy'] == accuracy)]

            #         sns.kdeplot(simulated_rt_data['simulated_rts'],
            #                     label='simulated reaction times' if 'simulated reaction times' not in labels_added else '_nolegend_',
            #                     ax=ax, color='#CC79A7')
            #         sns.kdeplot(experimental_rt_data['experimental_rts'],
            #                     label='experimental reaction times' if 'experimental reaction times' not in labels_added else '_nolegend_',
            #                     ax=ax, color='#0072B2')

            #         labels_added.update(['simulated reaction times', 'experimental reaction times'])

            #         ax.annotate(f"{congruency.capitalize()}, {'Correct' if accuracy else 'Incorrect'}",
            #                     xy=(0.96, 1), xycoords='axes fraction', xytext=(0, -5),
            #                     textcoords='offset points', fontsize='small', ha='right', va='top',
            #                     bbox=dict(facecolor='white', edgecolor='none', pad=3.0))
            #         ax.annotate(f"Experimental N = {len(experimental_rt_data)}",
            #                     xy=(0.96, 0.9), xycoords='axes fraction', xytext=(0, -5),
            #                     textcoords='offset points', fontsize='small', ha='right', va='top',
            #                     bbox=dict(facecolor='white', edgecolor='none', pad=3.0))
            #         ax.annotate(f"Simulated N = {len(simulated_rt_data)}",
            #                     xy=(0.96, 0.8), xycoords='axes fraction', xytext=(0, -5),
            #                     textcoords='offset points', fontsize='small', ha='right', va='top',
            #                     bbox=dict(facecolor='white', edgecolor='none', pad=3.0))
            #         ax.set_xlabel("Response Time (s)")

            #     fig.suptitle(f'Posterior Predictive Check Participant {id}')
            #     plt.tight_layout(rect=[0, 0, 1, 0.95])
            #     fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.94), ncol=1,
            #                bbox_transform=fig.transFigure, fontsize='x-small', frameon=False)
            #     plt.show()

            #     plot_path = os.path.join(posterior_predictive_check_model_dir, f"participant_{id}.png")
            #     fig.savefig(plot_path, dpi=400)

            if not return_dataframes:
                df.to_csv(os.path.join(fit_parameters_dir, f"{model.__class__.__name__}_{output_fileName}"), index=False)

        if return_dataframes:
            dflist.append(df)

        df.to_csv(os.path.join(fit_parameters_dir, f"{model.__class__.__name__}_{output_fileName}"), index=False)

    return dflist if return_dataframes else None