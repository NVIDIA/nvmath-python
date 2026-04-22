{# This template replaces the default autosummary Sphinx template

   - The default template does not show all methods and attributes on one page.
   - Attributes are listed before methods
   - __init__ is skipped because the function signature is already shown
#}
{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: {{ _('Attributes') }}

   {% for item in attributes %}
   .. autoattribute:: {{ item }}
   {%- endfor %}

   {% endif %}
   {% endblock %}

   {% block methods %}
   {% if methods %}
   .. rubric:: {{ _('Methods') }}

   {% for item in methods %}
   {% if item != '__init__' %}
   .. automethod:: {{ item }}
   {% endif %}
   {%- endfor %}

   {% endif %}
   {% endblock %}
