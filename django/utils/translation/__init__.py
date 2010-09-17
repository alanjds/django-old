"""
Internationalization support.
"""
from django.utils.encoding import force_unicode
from django.utils.functional import lazy, curry


__all__ = ['gettext', 'gettext_noop', 'gettext_lazy', 'ngettext',
        'ngettext_lazy', 'string_concat', 'activate', 'deactivate',
        'get_language', 'get_language_bidi', 'get_date_formats',
        'get_partial_date_formats', 'check_for_language', 'to_locale',
        'get_language_from_request', 'templatize', 'ugettext', 'ugettext_lazy',
        'ungettext', 'deactivate_all']

# Here be dragons, so a short explanation of the logic won't hurt:
# We are trying to solve two problems: (1) access settings, in particular
# settings.USE_I18N, as late as possible, so that modules can be imported
# without having to first configure Django, and (2) if some other code creates
# a reference to one of these functions, don't break that reference when we
# replace the functions with their real counterparts (once we do access the
# settings).

class TransProvider(object):
    """
    The purpose of this class is to store the actual translation function upon
    receiving the first call to that function. After this is done, changes to
    USE_I18N will have no effect to which function is served upon request. If
    your tests rely on changing USE_I18N, you can delete all the functions
    from _trans_provider.__dict__.
    
    Note that storing the function with setattr will have a noticeable
    performance effect, as access to the function goes the normal path,
    instead of using __getattr__.
    """
    def __getattr__(self, name):
        from django.conf import settings
        if settings.USE_I18N:
            from django.utils.translation import trans_real as trans_provider
        else:
            from django.utils.translation import trans_null as trans_provider
        setattr(self, name, getattr(trans_provider, name))
        return getattr(trans_provider, name)

_trans_provider = TransProvider()

# The TransProvider class is no more needed, so remove it from the namespace.
del TransProvider

def gettext_noop(message):
    global _trans_provider
    return _trans_provider.gettext_noop(message)

ugettext_noop = gettext_noop

def gettext(message):
    global _trans_provider
    return _trans_provider.gettext(message)

def ngettext(singular, plural, number):
    global _trans_provider
    return _trans_provider.ngettext(singular, plural, number)

def ugettext(message):
    global _trans_provider
    return _trans_provider.ugettext(message)

def ungettext(singular, plural, number):
    global _trans_provider
    return _trans_provider.ungettext(singular, plural, number)

ngettext_lazy = lazy(ngettext, str)
gettext_lazy = lazy(gettext, str)
ungettext_lazy = lazy(ungettext, unicode)
ugettext_lazy = lazy(ugettext, unicode)

def activate(language):
    global _trans_provider
    return _trans_provider.activate(language)

def deactivate():
    global _trans_provider
    return _trans_provider.deactivate()

def get_language():
    global _trans_provider
    return _trans_provider.get_language()

def get_language_bidi():
    global _trans_provider
    return _trans_provider.get_language_bidi()

def get_date_formats():
    global _trans_provider
    return _trans_provider.get_date_formats()

def get_partial_date_formats():
    global _trans_provider
    return _trans_provider.get_partial_date_formats()

def check_for_language(lang_code):
    global _trans_provider
    return _trans_provider.check_for_language(lang_code)

def to_locale(language):
    global _trans_provider
    return _trans_provider.to_locale(language)

def get_language_from_request(request):
    global _trans_provider
    return _trans_provider.get_language_from_request(request)

def templatize(src):
    global _trans_provider
    return _trans_provider.templatize(src)

def deactivate_all():
    global _trans_provider
    return _trans_provider.deactivate_all()

def _string_concat(*strings):
    """
    Lazy variant of string concatenation, needed for translations that are
    constructed from multiple parts.
    """
    return u''.join([force_unicode(s) for s in strings])
string_concat = lazy(_string_concat, unicode)
