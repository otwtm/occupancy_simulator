# noinspection PyInterpreter
from flask import render_template, flash, redirect, url_for, send_file, Response, session
from app import app, db
from app.forms import InfoForm, FormEntry, MyForm
from app.myfunctions import make_figures, create_csv_file, make_profiles, make_household, id_generator
import random

import numpy as np
from io import BytesIO
from datetime import datetime, timedelta
from app.models import Person, Bedroom

from flask import send_file
import pandas as pd
from bokeh.embed import components
from app.simulate import simulate

import pandas as pd
import numpy as np
import jsonpickle

from wtforms import SelectField, SubmitField, IntegerField, FieldList, FormField
from flask import send_file, make_response


@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
    global resp
    error = None
    form = InfoForm()

    if form.validate_on_submit():
        session['number_adults'] = form.number_adults_field.data
        session['number_children'] = form.number_children_field.data
        session['number_bedrooms'] = form.number_bedrooms_field.data
        session['startdate'] = datetime.strftime(form.startdate_field.data, '%Y-%m-%d')
        session['enddate'] = datetime.strftime(form.enddate_field.data, '%Y-%m-%d')
        return redirect(url_for('allocate_bedrooms'))
    return render_template('index.html', title='Home', form=form, error=error)



@app.route('/allocate_bedrooms', methods=['GET', 'POST'])
def allocate_bedrooms():
    bedrooms = []
    for i in range(session['number_bedrooms']):
        bedrooms.append(Bedroom(id=i, name='Bedroom {}'.format(i+1)))

    persons = make_household(number_adults=session['number_adults'], number_children=session['number_children'])

    form = MyForm(form_entries=[{} for i in range(len(persons))])
    for i in range(len(persons)):
        form.form_entries[i].selectfield.choices = [(bedroom.id, bedroom.name) for bedroom in bedrooms]
        form.form_entries[i].selectfield.label.text = persons[i].name
    #flash(form.errors)

    if form.validate_on_submit():
        for i in range(len(persons)):
            persons[i].bedroom = form.form_entries[i].selectfield.data
        persons_json = [jsonpickle.encode(person) for person in persons]
        session['persons'] = persons_json

        return redirect(url_for('present_sim'))
    return render_template('allocate_bedrooms.html', title='Allocate bedrooms', form=form)


@app.route('/present_sim', methods=['GET', 'POST'])
def present_sim():
    global resp
    persons = [jsonpickle.decode(person) for person in session['persons']]
    startdate = datetime.strptime(session['startdate'], '%Y-%m-%d')
    enddate = datetime.strptime(session['enddate'], '%Y-%m-%d')
    room_profiles = make_profiles(persons, startdate=startdate, enddate=enddate, number_bedrooms=session['number_bedrooms'])

    time_axis = pd.date_range(start=startdate, end=(enddate+timedelta(minutes=1430)), freq='10min')

    profile_dict = {'datetime': time_axis, 'Kitchen': room_profiles[0], 'Bathroom': room_profiles[1], 'Livingroom': room_profiles[2]}
    for i in range(session['number_bedrooms']):
        profile_dict.update({'Bedroom {}'.format(i+1): room_profiles[3+i]})

    session['session_key'] = id_generator()
    df = pd.DataFrame(profile_dict)
    df.to_sql(name='occupancy_profiles'+session['session_key'], con=db.engine, index=False, if_exists="replace")

    plots = make_figures(df, number_bedrooms=session['number_bedrooms'])
    print("hello")
    plot_elements = [components(p) for p in plots]
    print("world")

    return render_template('present_sim.html', title='Present simulation', plot_elements=plot_elements)


@app.route('/return_csv')
def return_csv():
    table_name = 'occupancy_profiles'+session['session_key']
    df = pd.read_sql_query("select * from {};".format(table_name), con=db.engine)
    db.engine.execute('DROP TABLE IF EXISTS {};'.format(table_name))
    resp = create_csv_file(df)
    return resp


