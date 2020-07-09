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

      
django_names = {'Variable_Dil1_Entrada':'da_dil1',
                'Variable_AGV_Entrada': 'da_agv_in',
                'Variable_DQO_Entrada': 'da_dqo_in',
                'Variable_Biomasa_Salida':'da_biomasa_x',
                'Variable_DQO_Salida':'da_dqo_out',
                'Variable_AGV_Salida':'da_agv_out',
                'Variable_Ace_Entrada':'mec_agv_in',
                'Variable_Dil2_Entrada': 'mec_dil2',
                'Variable_Eapp_Entrada': 'mec_eapp',
                'Variable_Ace_Salida': 'mec_ace',
                'Variable_xa_Salida': 'mec_xa',
                'Variable_xm_Salida': 'mec_xm',
                'Variable_xh_Salida': 'mec_xh',
                'Variable_mox_Salida': 'mec_mox',
                'Variable_imec_Salida': 'mec_imec',
                'Variable_QH2_Salida': 'mec_qh2',
                }


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


def get_top_status(proceso):
    try:
        estados = proceso.tieneEstado
        if len(estados) > 1:
            for e in estados:
                if "Falla" == e.get_name():
                    return e.get_name()
                elif "Riesgo" == e.get_name():
                    return e.get_name()
    except NotObjectPropertyFound as e:
        print(f"ObjectProperty:tieneEstado not found for {proceso}, {e}"
              )
    return proceso.tieneEstado[0].get_name()


def merge_mensaje(mensaje):
    merged_mensaje = ''
    for m in mensaje:
        if type(m) is str:
            merged_mensaje += m + '.\n'
        else:
            return "Onto-Error: mensaje is not string"
    return merged_mensaje


def build_dict(seq, key):
    """  indexing the regla list  by storid (using a dictionary),
    this way 'get' operations would be O(1) time"""
    return dict((d[key], dict(d, index=index)) for (index, d) in enumerate(seq))


def clean_name(ind_name):
    """ Used for tree graph"""
    replace_list = ["Error_", "Variable_", ]
    for word in replace_list:
        if word in ind_name:
            return ind_name.replace(word, "")
    return ind_name


def get_individual_object_properties(ind):
    list_properties = []
    # Get all Object Properties for the current individual (ind)
    for p in ind.get_properties():
        if isinstance(p, ObjectPropertyClass):
            list_properties.append(p)
    return list_properties


def get_tree_data_structure(onto=None):
    value = []
    paths = []
    # print(onto)
    dict_estados = {'Normal': 0, 'Falla': 1, 'Riesgo': 2, 'Alto': 3, 'Bajo': 4}
    separation = '.'

    if onto:
        onto = onto
    else:
        """ Load default onto"""
        new_world = owlready2.World()
        # Loading our ontologia
        onto = new_world.get_ontology(onto_dir_path).load()
        sync_reasoner_pellet(onto, infer_data_property_values=True, infer_property_values=True, debug=0)
        print("Using default onto")

    procesos = onto.search(type=onto.Proceso)
    # print(f"ONTO_PARSER: {onto}")
    # print(f"ONTO_PARSER: {onto.search(type=onto.Biorrefineria)}")
    root = onto.search(type=onto.Biorrefineria)[0].get_name()  # Biorrefineria_1
    paths.append(root)  # Root
    value.append(0)

    for proceso in procesos:
        path_lvl_1 = ''
        path_lvl_1 += root + separation + proceso.get_name()  # path = 'Biorrefineria.Proceso'
        paths.append(path_lvl_1)
        estado = get_top_status(proceso)
        # print(f'ontoParser-estado: {estado}')
        value.append(dict_estados[estado])  # value according state
        prop_lvl_1 = get_individual_object_properties(proceso)
        #        print(prop_lvl_1)
        prop_filtered = [p for p in prop_lvl_1 if
                         (p.get_name().startswith(('presentaError', 'tieneEstado', 'enRiesgoDePresentar')))]

        for p1 in prop_filtered:
            if p1.get_name() == 'tieneEstado':
                path_lvl_2 = ''
                path_lvl_2 += separation + p1.get_name()  # '.prop1' -> '.tieneEstado'
                paths.append(path_lvl_1 + path_lvl_2)  # 'Biorrefineria.tieneEstado
                # print(f"inside 'tieneEstado': {path_lvl_1 + path_lvl_2}")
                value.append(0)
                ind_lvl_2 = getattr(proceso,
                                    p1.get_name())  # get individuals which have a p object property link with Proceso
                for ind in ind_lvl_2:
                    path_lvl_3 = ''
                    ind_name = ind.get_name()
                    path_lvl_3 += separation + ind_name  # '.prop1.ind2' -> '.tieneEstado.Anormal'
                    paths.append(path_lvl_1 + path_lvl_2 + path_lvl_3)
                    # print(f"inside 'tieneEstado': {path_lvl_1 + path_lvl_2}")
                    value.append(dict_estados[ind_name])  # value according state
            else:
                path_lvl_2 = ''
                path_lvl_2 += separation + p1.get_name()  # '.prop1' -> '.presentaError'
                paths.append(path_lvl_1 + path_lvl_2)  # 'Biorrefineria.Proceso.presentaError
                value.append(0)
                ind_lvl_2 = getattr(proceso,
                                    p1.get_name())  # get individuals which have a p object property link with Proceso

                for ind in ind_lvl_2:
                    ind_name = ind.get_name()  # 'Error_Ace_Salida_Alto'
                    path_lvl_3 = ''
                    path_lvl_3 += separation + clean_name(
                        ind_name)  # '/prop1/ind2/' -> 'presentaError/Ace_Salida_Alto/'
                    paths.append(
                        path_lvl_1 + path_lvl_2 + path_lvl_3)  # 'Biorrefineria.presentaError.Error_Ace_Salida_Alto
                    value.append(20)
                    prop_lvl_3 = get_individual_object_properties(ind)
                    prop_lvl_3_filtered = [p for p in prop_lvl_3 if (
                        p.get_name().startswith(('afectaVariable',)))]

                    for p2 in prop_lvl_3_filtered:
                        path_lvl_4 = ''
                        path_lvl_4 += separation + p2.get_name()  # '.prop2' -> '.afectaVariable'
                        paths.append(
                            path_lvl_1 + path_lvl_2 + path_lvl_3 + path_lvl_4)  # 'Biorrefineria.presentaError.Error_Ace_Salida_Alto.afectaVariable
                        value.append(0)
                        ind_lvl_4 = getattr(ind, p2.get_name())

                        for ind_4 in ind_lvl_4:
                            ind_name = ind_4.get_name()
                            path_lvl_5 = ''
                            path_lvl_5 += separation + clean_name(ind_name)

                            paths.append(path_lvl_1 + path_lvl_2 + path_lvl_3 + path_lvl_4 + path_lvl_5)
                            value.append(ind.tieneValorCensado if ind.tieneValorCensado else 0)

    paths = pd.DataFrame(paths)
    paths['value'] = value
    paths.columns = ['id', 'value']
    return paths.to_json(orient='records')


def clean_rule(regla):
    new_world = owlready2.World()
    # Loading our ontologia
    onto_src = new_world.get_ontology(onto_dir_path).load()
    # Deleting onto name and change '^' by ','
    print(f"onto_src.name = {onto_src.name}")
    clean_regla = regla.replace(onto_src.name + ".", '')
    clean_regla = clean_regla.replace('^', ',')
    clean_regla = clean_regla.replace('swrlb:', '')
    clean_regla = clean_regla.replace('biorrefineria:', '')
    return clean_regla


def get_iri(name):
    new_world = owlready2.World()
    # Loading our ontologia
    onto_src = new_world.get_ontology(onto_dir_path).load()
    return onto_src.base_iri + name


def get_list_of_procesos():
    new_world = owlready2.World()
    # Loading our ontologia
    onto_src = new_world.get_ontology(onto_dir_path).load()
    all_procesos = onto_src.search(type=onto_src.Proceso)
    proceso_list = []

    for p in all_procesos:
        iri = p.get_iri()
        nombre = p.tieneNombre
        descripcion = p.tieneDescripcion
        alimenta_proceso = [proceso.tieneNombre for proceso in p.alimentaProceso] if p.alimentaProceso else ""

        proceso_list.append({'iri': iri,
                             'nombre': nombre,
                             'descripcion': descripcion,
                             'alimenta_proceso': alimenta_proceso})
    return proceso_list


def get_list_of_variables():
    new_world = owlready2.World()
    # Loading our ontologia
    onto_src = new_world.get_ontology(onto_dir_path).load()
    sync_reasoner_pellet(onto_src, infer_data_property_values=True, infer_property_values=True, debug=0)
    all_variables = onto_src.search(type=onto_src.Variable)
    variable_list = []

    for v in all_variables:
        # nombre = str(v).split('.')[1]
        nombre = v.get_name()
        descripcion = v.tieneDescripcion if v.tieneDescripcion else "No definido"
        unidad = str(v.tieneUnidadDeMedida) if v.tieneUnidadDeMedida else "No definido"
        maximo = v.tieneValorMaximo if v.tieneValorMaximo else 0.0
        minimo = v.tieneValorMinimo if v.tieneValorMinimo else 0.0
        valor_riesgo_maximo = v.tieneValorRiesgoMaximo[0] if v.tieneValorRiesgoMaximo else 0.0
        valor_riesgo_minimo = v.tieneValorRiesgoMinimo[0] if v.tieneValorRiesgoMinimo else 0.0
        nominal = v.tieneValorNominal if v.tieneValorNominal else 0.0
        # print(getattr(v, "esVariableDe")[0].get_name().split('_', 1)[1])
        proceso = getattr(v, "esVariableDe")[0].get_name().split('_', 1)[1] if v.esVariableDe else "No definido"

        variable_list.append({'nombre': nombre,
                              'iri': v.iri,
                              'descripcion': descripcion,
                              'unidad': unidad,
                              'maximo': maximo,
                              'minimo': minimo,
                              'rmax': valor_riesgo_maximo,
                              'rmin': valor_riesgo_minimo,
                              'nominal': nominal,
                              'proceso': proceso})
    return variable_list


def get_list_of_errors():
    new_world = owlready2.World()
    # Loading our ontologia
    onto_src = new_world.get_ontology(onto_dir_path).load()
    """To obtain a list of all errors in the ontologia and its recomendacion"""
    all_errors = onto_src.search(type=onto_src.Error)
    errors_list = []

    for e in all_errors:
        nombre = e.get_name()
        descripcion = str(e.tieneDescripcion) if e.tieneDescripcion else "No definido"
        peligro = str(e.tieneNivelDePeligro[0]) if e.tieneNivelDePeligro else "No definido"
        variable = [var.get_name() for var in e.afectaVariable] if e.afectaVariable else "No definido"
        recomendacion = getattr(e, "tieneRecomendacion")[0].get_name() if e.tieneRecomendacion else "No definido"
        r_iri = str(e.tieneRecomendacion[0].iri) if e.tieneRecomendacion else "No definido"
        es_error_de = str(e.esErrorDe[0]) if e.esErrorDe else "No definido"
        desc_recomendacion = str(e.tieneRecomendacion[0].tieneDescripcion[0]) if e.tieneRecomendacion and \
                                                                                                e.tieneRecomendacion[
                                                                                                    0].tieneDescripcion else "No definido"
        print(r_iri)
        errors_list.append({'nombre': nombre,
                            'iri': e.iri,
                            'descripcion': descripcion,
                            'peligro': peligro,
                            'variable': variable,
                            'recomendacion': recomendacion,
                            'r_iri': r_iri,
                            'desc_recomendacion': desc_recomendacion,
                            'es_error_de': es_error_de,})
    return errors_list


def get_list_of_rules():
    new_world = owlready2.World()
    # Loading our ontologia
    onto_src = new_world.get_ontology(onto_dir_path).load()
    # from operator import itemgetter
    # # Obtain rules from the ontologia
    # # Loading our ontologia
    rules = list(onto_src.rules())
    rules_list = []

    for r in rules:
        rules_list.append({'nombre': r.label[0] if r.label else "No definido",
                           'proceso': r.label[0].split('_', 1)[0] if r.label else "No definido",
                           'descripcion': r.comment[0] if r.comment else "No definido",
                           'regla': str(r),
                           'activo': r.isRuleEnabled})
    return rules_list


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
    onto = new_world.get_ontology(onto_dir_path).load()
    # Creating individuals of Lectura that will be used by the rules
    onto.Variable_Dil1_Entrada.tieneValorPropuesto = float(data.da_dil)
    onto.Lectura_AGV_Entrada.tieneValorCensado = float(data.da_agv_in)
    onto.Lectura_DQO_Entrada.tieneValorCensado = float(data.da_dqo_in)
    onto.Lectura_Biomasa_Salida.tieneValorCensado = float(data.da_biomasa)
    onto.Lectura_DQO_Salida.tieneValorCensado = float(data.da_dqo_out)
    onto.Lectura_AGV_Salida.tieneValorCensado = float(data.da_agv_out)
    onto.Variable_Dil2_Entrada.tieneValorPropuesto = float(data.mec_dil)
    onto.Lectura_Ace_Salida.tieneValorCensado = float(data.mec_ace_out)
    onto.Lectura_xa_Salida.tieneValorCensado = float(data.mec_xa)
    onto.Lectura_xm_Salida.tieneValorCensado = float(data.mec_xm)
    onto.Lectura_xh_Salida.tieneValorCensado = float(data.mec_xh)
    onto.Lectura_mox_Salida.tieneValorCensado = float(data.mec_mox)
    onto.Lectura_imec_Salida.tieneValorCensado = float(data.mec_imec)
    onto.Lectura_QH2_Salida.tieneValorCensado = float(data.mec_qh2)

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


def onto_create_error(error):
    new_world = owlready2.World()
    # Loading our ontologia
    onto_src = new_world.get_ontology(onto_dir_path).load()
    error = json.loads(error)
    print(error)
    # for v in error.variable.all():
    #     afecta_variables.append(v)
    e = onto_src.Error(error['nombre'],
                       namespace=onto_src,
                       tieneDescripcion=error['descripcion'],
                       tieneFechaDeActualizacion=[error['created_at']],
                       tieneNivelDePeligro=[error['peligro']],
                       tieneNombre=error['nombre'],
                       esErrorDe=[error['es_error_de']],)
    # print( error.variable.all())
    for v in error['variables']:
        # print(v)
        print(f'Adding {onto_src[v]} to {e}')
        e.afectaVariable.append(onto_src[v])

    onto_src.save(onto_dir_path, format="rdfxml")
    print(f'Ontology: Error {e} created')


def onto_update_error(old_nombre, updated_error, fields_updated):
    new_world = owlready2.World()
    # Loading our ontologia
    onto_src = new_world.get_ontology(onto_dir_path).load()
    print(f"old_nombre: {old_nombre}")
    print(f"fields_updated: {fields_updated}")
    print(f"updated_error: {updated_error}")

    error = json.loads(updated_error)
    nombre = error['nombre']
    descripcion = error['descripcion']
    peligro = error['peligro']
    es_error_de = error['es_error_de']
    variables = error['variables']
    date = error['updated_at']

    error_onto = onto_src[nombre]

    if 'nombre' in fields_updated:
        error_onto = onto_src[old_nombre]
        error_onto.iri = onto_src.base_iri + nombre  # Change old iri name for the new one
        error_onto.tieneNombre = nombre
        print(f"Nombre updated for error {error_onto}")

    # Update variables affected
    if 'variable' in fields_updated:
        variables_from_request = [onto_src[var] for var in variables]
        error_onto.afectaVariable.clear()
        print("List cleared")
        error_onto.afectaVariable = variables_from_request
        print("New elements added")

    error_onto.tieneDescripcion = descripcion
    error_onto.tieneNivelDePeligro = [peligro]
    error_onto.esErrorDe = [es_error_de]
    error_onto.tieneFechaDeActualizacion = [date]
    onto_src.save(onto_dir_path, format="rdfxml")
    print(f'Ontology: Error {error_onto} updated')


def onto_create_recomendacion(recomendacion: json):
    new_world = owlready2.World()
    onto_src = new_world.get_ontology(onto_dir_path).load()
    recomendacion = json.loads(recomendacion)
    print(recomendacion)

    r = onto_src.Recomendacion(recomendacion['nombre'],
                               namespace=onto_src,
                               tieneDescripcion=recomendacion['descripcion'],
                               tieneFechaDeActualizacion=[recomendacion['created_at']],
                               tieneNombre=recomendacion['nombre'],)

    for err in recomendacion['errores']:
        # print(err)
        print(f'Adding {r} as a recommendation of  {onto_src[err]} ')
        r.esRecomendacionDe.append(onto_src[err])

    onto_src.save(onto_dir_path, format="rdfxml")
    print(f'Ontology: Recomendacion {r} created')


def onto_update_recomendacion(recomendacion, fields_updated, old_name=None):
    new_world = owlready2.World()
    onto_src = new_world.get_ontology(onto_dir_path).load()
    recomendacion = json.loads(recomendacion)
    print(f"Inside ONTO_UPDATE_RECOMENDACION")

    print(f"recomendacion: {recomendacion}")
    print(f"old_name: {old_name}")

    print(f"fields_updated: {fields_updated}")

    nombre = recomendacion['nombre']
    descripcion = recomendacion['descripcion']
    errores = recomendacion['errores']
    recomendacion_iri = recomendacion['iri'].split("#")[1]
    date = recomendacion['updated_at']

    recomendacion_onto = onto_src[nombre]

    if 'nombre' in fields_updated:
        recomendacion_onto = onto_src[old_name]
        print(f"old_recomendacion_onto: {recomendacion_onto}")
        recomendacion_onto.iri = onto_src.base_iri + nombre # Change old iri name for the new one
        recomendacion_onto.tieneNombre = nombre
        print(f"Nombre updated for recomendacion {recomendacion_onto}")

    # Update the Error to which belongs the current recomendacion
    if 'error' in fields_updated:
        errores_from_request = [onto_src[err] for err in errores]
        print(f"Error_form_request: {errores_from_request}")
        recomendacion_onto.esRecomendacionDe.clear()
        print("List cleared")
        recomendacion_onto.esRecomendacionDe = errores_from_request
        print("New elements added")

    # update Error coming from ErrorCreateView
    if 'error_from_createview' in fields_updated:
        for err in errores:
            print(f"err: {err}")
            print(f"Adding error {err} to {recomendacion_onto}")
            print(f"onto_error_name: {onto_src[err]}")
            print(f"esRecomendacionDe: {recomendacion_onto.esRecomendacionDe}")
            if onto_src[err]:
                if onto_src[err] in recomendacion_onto.esRecomendacionDe:
                    print(f"{err} is already added")
                else:
                    recomendacion_onto.esRecomendacionDe.append(onto_src[err])
            else:
                print(f"Error-No storeid for {err}")
        print("New elements added from ErrorCreateView")

    recomendacion_onto.tieneDescripcion = descripcion
    recomendacion_onto.tieneFechaDeActualizacion = [date]

    onto_src.save(onto_dir_path, format="rdfxml")
    print(f'Ontology: recomendacion {recomendacion_onto} updated')


def onto_update_regla(regla, old_name):
    new_world = owlready2.World()
    # Loading our ontologia
    onto_src = new_world.get_ontology(onto_dir_path).load()
    regla = json.loads(regla)
    print(f"regla_loads: {regla}")

    list_of_rules = get_list_of_rules()
    dic_of_rules = build_dict(list_of_rules, key='nombre')
    rules = list(onto_src.rules())
    rule_position = dic_of_rules.get(old_name)['index']
    r = rules[rule_position]
    print("rule which will be changed is {}".format(old_name))
    print("rule in dict rule position {} is {}".format(rule_position, dic_of_rules.get(old_name)))
    print("rule in rule position {} is {}".format(rule_position, r.label))
    # print(f"rule_before_clean: {regla['swrl']}")
    # print(f"rule_after_clean: {clean_rule(regla['swrl'])}")

    try:
        print(f"updating label - {r.label}")
        r.label = regla['nombre']  # Change old  name for the new one
        print(f"label updated - {r.label}")
        r.comment = regla['descripcion']
        print(f"comment updated - {regla['descripcion']}")
        r.isRuleEnabled = [regla['activo']]
        print(f"isRuleEnabled updated - {regla['activo']}")

        # r.set_as_rule(clean_rule(regla['swrl']))
        # print(f"set_as_rule updated")
    except Exception as e:
        print("Error updating rule")
        print(e)

    print("Regla updated")
    onto_src.save(onto_dir_path, format="rdfxml")

def onto_delete_individual(iri):
    new_world = owlready2.World()
    # Loading our ontologia
    onto_src = new_world.get_ontology(onto_dir_path).load()
    nombre = iri.split("#")[1]
    print(f'Deleting {nombre}')
    individual_onto = onto_src[nombre]
    try:
        destroy_entity(individual_onto)
        print(f'Ontology: individual {individual_onto} deleted')
        onto_src.save(onto_dir_path, format="rdfxml")
    except Exception as e:
        print(f'Individual {nombre} does not exist')
        print(f'Error {e}')

# Too unstable - it broke the ontology
# def onto_delete_regla(nombre):
#     new_world = owlready2.World()
#     # Loading our ontologia
#     onto_src = new_world.get_ontology(onto_dir_path).load()
#     rules = list(onto_src.rules())
#     list_of_rules = get_list_of_rules()
#     print(f'list_of_rules: {list_of_rules}')
#     position = next((index for (index, d) in enumerate(list_of_rules) if d["nombre"] == nombre), None)
#     print(f'position: {position}')
#
#     try:
#         regla = rules[position]
#         print(f'rules[position]: {rules[position]}')
#         print(f'Deleting {nombre}')
#         destroy_entity(regla, undoable=True)
#         print(f'Ontology: Regla {regla} deleted')
#         onto_src.save(onto_dir_path, format="rdfxml")
#         print(f'Ontology: regla {regla} deleted')
#     except Exception as e:
#         print(f'Regla does not exist')
#         print(f'Error {e}')

# Too unstable - it broke the ontology
#
# def onto_create_regla(regla: json) -> str:
#     new_world = owlready2.World()
#     onto_src = new_world.get_ontology(onto_dir_path).load()
#     regla = json.loads(regla)
#     with onto_src:
#         r = Imp()  # Create a new instance of rule
#         try:
#             r.set_as_rule(clean_rule(regla['swrl']))  # Setting the cleaned rule
#         except Exception as e:
#             print(f'ERROR: BAD RULE FORMAT {e}')
#             return 'ERROR: RULE NOT IN SWRL FORMAT'
#         r.label = [regla['nombre']]
#         r.comment = [regla['descripcion'] if regla['descripcion'] != '' else 'No definido']
#         r.isRuleEnabled = [regla['activo']]  # To enable the new rule
#     onto_src.save(onto_dir_path, format="rdfxml")
#     print("ONTO: Rule created")
#     return "OK"