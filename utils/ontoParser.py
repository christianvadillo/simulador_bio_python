import json
import pandas as pd
from owlready2 import *
from datetime import datetime

#from custom_exceptions import NotObjectPropertyFound

# from datetime import datetime
# onto_dir_path = "static/owl_files/bio_feb_7_2020.owl" Working
# onto_dir_path = "static/owl_files/bio_mar_7_2020_10am.owl" Working
onto_dir_path = "models/bio_abril_10_2020_10am_testing_updates.owl"
# onto_src = get_ontology(onto_dir_path).load()
print("###################### ONTO LOADED ######################")

def onto_get_var_limits():
    da_vars = {}
    mec_vars = {}
    # creating a new world to isolate the reasoning results
    new_world = owlready2.World()
    # Loading our ontologia
    onto = new_world.get_ontology(onto_dir_path).load()
    
    variables = onto.search(type=onto.Variable)
    for var in variables:
        try:
            if 'DA' in var.esVariableDe[0].get_name():
                da_vars[django_names[var.get_name()]] = {'min': var.tieneValorMinimo,
                                           'max': var.tieneValorMaximo}
#                print(var.esVariableDe[0].get_name())
            if 'MEC' in var.esVariableDe[0].get_name():
                mec_vars[django_names[var.get_name()]] = {'min': var.tieneValorMinimo,
                                           'max': var.tieneValorMaximo}
#                print(var.esVariableDe[0].get_name())
        except Exception as e:
            print("None", e)
    return da_vars, mec_vars


def update_onto_limits(var_boundaries):
#    print("Updating boundaries")
#    print(var_boundaries)
    # creating a new world to isolate the reasoning results
    new_world = owlready2.World()
    # Loading our ontologia
    onto = new_world.get_ontology(onto_dir_path).load()

    # Updating DA variables
    onto.Variable_Dil1_Entrada.tieneValorMinimo = float(var_boundaries.loc['min']['da_dil1'])
    onto.Variable_AGV_Entrada.tieneValorMinimo = float(var_boundaries.loc['min']['da_agv_in'])
    onto.Variable_DQO_Entrada.tieneValorMinimo = float(var_boundaries.loc['min']['da_dqo_in'])
    onto.Variable_Biomasa_Salida.tieneValorMinimo = float(var_boundaries.loc['min']['da_biomasa_x'])
    onto.Variable_DQO_Salida.tieneValorMinimo = float(var_boundaries.loc['min']['da_dqo_out'])
    onto.Variable_AGV_Salida.tieneValorMinimo = float(var_boundaries.loc['min']['da_agv_out'])
    onto.Variable_Dil1_Entrada.tieneValorMaximo = float(var_boundaries.loc['max']['da_dil1'])
    onto.Variable_AGV_Entrada.tieneValorMaximo = float(var_boundaries.loc['max']['da_agv_in'])
    onto.Variable_DQO_Entrada.tieneValorMaximo = float(var_boundaries.loc['max']['da_dqo_in'])
    onto.Variable_Biomasa_Salida.tieneValorMaximo = float(var_boundaries.loc['max']['da_biomasa_x'])
    onto.Variable_DQO_Salida.tieneValorMaximo = float(var_boundaries.loc['max']['da_dqo_out'])
    onto.Variable_AGV_Salida.tieneValorMaximo = float(var_boundaries.loc['max']['da_agv_out'])

    # Updating MEC variables
    onto.Variable_Dil2_Entrada.tieneValorMinimo = float(var_boundaries.loc['min']['mec_dil2'])
    onto.Variable_Eapp_Entrada.tieneValorMinimo = float(var_boundaries.loc['min']['mec_eapp'])
    onto.Variable_Ace_Salida.tieneValorMinimo = float(var_boundaries.loc['min']['mec_ace'])
    onto.Variable_xa_Salida.tieneValorMinimo = float(var_boundaries.loc['min']['mec_xa'])
    onto.Variable_xm_Salida.tieneValorMinimo = float(var_boundaries.loc['min']['mec_xm'])
    onto.Variable_xh_Salida.tieneValorMinimo = float(var_boundaries.loc['min']['mec_xh'])
    onto.Variable_mox_Salida.tieneValorMinimo = float(var_boundaries.loc['min']['mec_mox'])
    onto.Variable_imec_Salida.tieneValorMinimo = float(var_boundaries.loc['min']['mec_imec'])
    onto.Variable_QH2_Salida.tieneValorMinimo = float(var_boundaries.loc['min']['mec_qh2'])
    onto.Variable_Dil2_Entrada.tieneValorMaximo = float(var_boundaries.loc['max']['mec_dil2'])
    onto.Variable_Eapp_Entrada.tieneValorMaximo = float(var_boundaries.loc['max']['mec_eapp'])
    onto.Variable_Ace_Salida.tieneValorMaximo = float(var_boundaries.loc['max']['mec_ace'])
    onto.Variable_xa_Salida.tieneValorMaximo = float(var_boundaries.loc['max']['mec_xa'])
    onto.Variable_xm_Salida.tieneValorMaximo = float(var_boundaries.loc['max']['mec_xm'])
    onto.Variable_xh_Salida.tieneValorMaximo = float(var_boundaries.loc['max']['mec_xh'])
    onto.Variable_mox_Salida.tieneValorMaximo = float(var_boundaries.loc['max']['mec_mox'])
    onto.Variable_imec_Salida.tieneValorMaximo = float(var_boundaries.loc['max']['mec_imec'])
    onto.Variable_QH2_Salida.tieneValorMaximo = float(var_boundaries.loc['max']['mec_qh2'])

    onto.save(onto_dir_path, format="rdfxml")

    print("limits updated")
    print()



def get_clean_individuals_list(proceso, property_):
    """ To obtain individuals without ontology path associated to the current state of the process
    through property_"""
    list_ = list(getattr(proceso, property_))
    ind_list = []
    # print(f"inside ontoParser-get_clean-{proceso}-{property_}- list_: {list_}")
    # print(f"inside ontoParser-get_clean-{proceso}-{property_}- ind_list: {ind_list}")
    for e in list_:
        # print(f"inside ontoParser-get_clean-{proceso}-{property_}- e: {e}")
        ind_list.append(e.get_name())
        # print(f"inside ontoParser-get_clean-{proceso}-{property_}: {ind_list}")

    return ind_list


def reasoner(data):
    # print("Inside OntoParser-Reasoner")
    # creating a new world to isolate the reasoning results
    new_world = owlready2.World()
    # Loading our ontologia
    # print(data)
    onto = new_world.get_ontology(onto_dir_path).load()
    # Creating individuals of Lectura that will be used by the rules
    onto.Variable_Dil1_Entrada.tieneValorPropuesto = float(data[0])
    onto.Lectura_AGV_Entrada.tieneValorCensado = float(data[1])
    onto.Lectura_DQO_Entrada.tieneValorCensado = float(data[2])
    onto.Lectura_Biomasa_Salida.tieneValorCensado = float(data[3])
    onto.Lectura_DQO_Salida.tieneValorCensado = float(data[4])
    onto.Lectura_AGV_Salida.tieneValorCensado = float(data[5])
    onto.Variable_Dil2_Entrada.tieneValorPropuesto = float(data[7])
    onto.Lectura_Ace_Salida.tieneValorCensado = float(data[8])
    onto.Lectura_xa_Salida.tieneValorCensado = float(data[9])
    onto.Lectura_xm_Salida.tieneValorCensado = float(data[10])
    onto.Lectura_xh_Salida.tieneValorCensado = float(data[11])
    onto.Lectura_mox_Salida.tieneValorCensado = float(data[12])
    onto.Lectura_imec_Salida.tieneValorCensado = float(data[13])
    onto.Lectura_QH2_Salida.tieneValorCensado = float(data[14])

    # Apply the rules using pellet reasoner
    sync_reasoner_pellet(onto, infer_data_property_values=True, infer_property_values=True, debug=0)

    # Obtain the list of states for each process
    # Transform list to dictionary
    processes = onto.search(type=onto.Proceso)
    processes_state = []

    for p in processes:
        has_status = [s.get_name() for s in list(p.tieneEstado)]
        # print(f"\nProceso: {p}")
        # print(f"estado: {has_status}")
        process_name = p.get_name().split('_', 1)[1]
        errores = get_clean_individuals_list(p, 'presentaError')
        riesgos = get_clean_individuals_list(p, 'enRiesgoDePresentar')
        mensaje = p.tieneDescripcionDeEstado

        processes_state.append({'proceso': process_name,
                                'estado': f'{has_status[0]}-{has_status[1]}' if len(has_status) == 2 else has_status[0],
                                'error': errores,
                                'riesgo': riesgos,
                                'mensaje': mensaje,
                                })
        # print(f"processes state: {processes_state}")
    return json.dumps(processes_state), onto
