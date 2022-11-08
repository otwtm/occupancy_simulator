from flask_wtf import FlaskForm
from wtforms import SelectField, SubmitField, IntegerField, FieldList, FormField
from wtforms.validators import DataRequired, NumberRange, ValidationError
from wtforms.fields import DateField #from wtforms.fields.html5 import DateField


class InfoForm(FlaskForm):
    number_adults_field = IntegerField('Number of adults', [NumberRange(min=1, max=10)])
    number_children_field = IntegerField('Number of children', [NumberRange(min=0, max=10)])
    number_bedrooms_field = IntegerField('Number of bedrooms')
    startdate_field = DateField('Start Date', format='%Y-%m-%d')
    enddate_field = DateField('End Date', format='%Y-%m-%d')
    submit_field = SubmitField('Next')

    def validate_enddate_field(form, field):
        if form.startdate_field.data is not None and field.data is not None\
                and field.data < form.startdate_field.data:
            raise ValidationError("End date must not be earlier than start date.")


class FormEntry(FlaskForm):
    selectfield = SelectField('Name', coerce=int)


class MyForm(FlaskForm):
    form_entries = FieldList(FormField(FormEntry))
    submit_field = SubmitField('Simulate')
