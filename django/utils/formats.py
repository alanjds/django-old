import decimal
import datetime

from django.conf import settings
from django.utils.translation import get_language, to_locale, check_for_language
from django.utils.importlib import import_module
from django.utils.encoding import smart_str
from django.utils import dateformat, numberformat, datetime_safe

def get_format_modules(reverse=False):
    """
    Returns an iterator over the format modules found in the project and Django
    """
    modules = []
    if not check_for_language(get_language()) or not settings.USE_L10N:
        return modules
    locale = to_locale(get_language())
    if settings.FORMAT_MODULE_PATH:
        format_locations = [settings.FORMAT_MODULE_PATH + '.%s']
    else:
        format_locations = []
    format_locations.append('django.conf.locale.%s')
    for location in format_locations:
        for l in (locale, locale.split('_')[0]):
            try:
                mod = import_module('.formats', location % l)
            except ImportError:
                pass
            else:
                # Don't return duplicates
                if mod not in modules:
                    modules.append(mod)
    if reverse:
        modules.reverse()
    return modules

def get_format(format_type):
    """
    For a specific format type, returns the format for the current
    language (locale), defaults to the format in the settings.
    format_type is the name of the format, e.g. 'DATE_FORMAT'
    """
    format_type = smart_str(format_type)
    if settings.USE_L10N:
        for module in get_format_modules():
            try:
                return getattr(module, format_type)
            except AttributeError:
                pass
    return getattr(settings, format_type)

def date_format(value, format=None):
    """
    Formats a datetime.date or datetime.datetime object using a
    localizable format
    """
    return dateformat.format(value, get_format(format or 'DATE_FORMAT'))

def time_format(value, format=None):
    """
    Formats a datetime.time object using a localizable format
    """
    return dateformat.time_format(value, get_format(format or 'TIME_FORMAT'))

def number_format(value, decimal_pos=None):
    """
    Formats a numeric value using localization settings
    """
    return numberformat.format(
        value,
        get_format('DECIMAL_SEPARATOR'),
        decimal_pos,
        get_format('NUMBER_GROUPING'),
        get_format('THOUSAND_SEPARATOR'),
    )

def localize(value):
    """
    Checks if value is a localizable type (date, number...) and returns it
    formatted as a string using current locale format
    """
    if isinstance(value, (decimal.Decimal, float, int)):
        return number_format(value)
    elif isinstance(value, datetime.datetime):
        return date_format(value, 'DATETIME_FORMAT')
    elif isinstance(value, datetime.date):
        return date_format(value)
    elif isinstance(value, datetime.time):
        return time_format(value, 'TIME_FORMAT')
    else:
        return value

def render_localize(value, context):
    if isinstance(value, (decimal.Decimal, float, int)):
        try:
            number_format = context.formats[('NUMBER_FORMAT', get_language())]
        except KeyError:
            number_format = (get_format('DECIMAL_SEPARATOR'),
                             get_format('NUMBER_GROUPING'),
                             get_format('THOUSAND_SEPARATOR'))
            context.formats[('NUMBER_FORMAT', get_language())] = number_format
            
        return (False, numberformat.format(
            value,
            number_format[0],
            None,
            number_format[1],
            number_format[2],
        ))
    elif isinstance(value, datetime.datetime):
        try:
            format = context.formats[('DATETIME_FORMAT', get_language())]
        except KeyError:
            format = get_format('DATETIME_FORMAT')
            context.formats[('DATETIME_FORMAT', get_language())] = smart_str(format)
        return (False, dateformat.format(value, format))
    elif isinstance(value, datetime.date):
        return (False, date_format(value))
    elif isinstance(value, datetime.time):
        return (True, time_format(value, 'TIME_FORMAT'))
    else:
        return (True, value)

def localize_input(value, default=None):
    """
    Checks if an input value is a localizable type and returns it
    formatted with the appropriate formatting string of the current locale.
    """
    if isinstance(value, (decimal.Decimal, float, int)):
        return number_format(value)
    if isinstance(value, datetime.datetime):
        value = datetime_safe.new_datetime(value)
        format = smart_str(default or get_format('DATETIME_INPUT_FORMATS')[0])
        return value.strftime(format)
    elif isinstance(value, datetime.date):
        value = datetime_safe.new_date(value)
        format = smart_str(default or get_format('DATE_INPUT_FORMATS')[0])
        return value.strftime(format)
    elif isinstance(value, datetime.time):
        format = smart_str(default or get_format('TIME_INPUT_FORMATS')[0])
        return value.strftime(format)
    return value

def sanitize_separators(value):
    """
    Sanitizes a value according to the current decimal and
    thousand separator setting. Used with form field input.
    """
    if settings.USE_L10N:
        decimal_separator = get_format('DECIMAL_SEPARATOR')
        if isinstance(value, basestring):
            parts = []
            if decimal_separator in value:
                value, decimals = value.split(decimal_separator, 1)
                parts.append(decimals)
            if settings.USE_THOUSAND_SEPARATOR:
                parts.append(value.replace(get_format('THOUSAND_SEPARATOR'), ''))
            else:
                parts.append(value)
            value = '.'.join(reversed(parts))
    return value
