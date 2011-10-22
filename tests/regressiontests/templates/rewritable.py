# -*- coding: utf-8 -*-
from django.template import Template, TemplateEncodingError, Context, loader
from django.utils.unittest import TestCase

class CSRFRewritableTest(TestCase):
    def test_single_template_positions(self):
        t = loader.get_template('csrf_base.html')
        c = Context({'csrf_token': 'TESTING'})
        resp = t.render(c)
        parts = resp.rewritable_parts['csrf_token']
        before_rewrite = resp.replace(u"'TESTING'", '')
        self.assertEquals(len(parts), 3)
        for start, end in parts:
            self.assertEquals(resp[start:end], u'TESTING')
        rewritten = resp.rewrite({'csrf_token':'NEW_TOKEN'})
        after_rewrite = rewritten.replace("'NEW_TOKEN'", '')
        self.assertEquals(before_rewrite, after_rewrite)
        self.assertTrue("'NEW_TOKEN'" in rewritten)
        self.assertFalse("'TESTING'" in rewritten)

    def test_extended_template_positions(self):
        t = loader.get_template('csrf_extended.html')
        c = Context({'csrf_token': 'TESTING', 'include_test': False})
        resp = t.render(c)
        parts = resp.rewritable_parts['csrf_token']
        self.assertEquals(len(parts), 4)
        pos = 0
        before_rewrite = resp.replace(u"'ANOT'", '').replace(u"'TESTING'", '')
        for start, end in parts:
            pos += 1 
            if pos == 2:
                self.assertEquals(resp[start:end], u'ANOT')
            else:
                self.assertEquals(resp[start:end], u'TESTING')
        rewritten = resp.rewrite({'csrf_token':'NEW_TOKEN'})
        after_rewrite = rewritten.replace("'NEW_TOKEN'", '')
        self.assertEquals(before_rewrite, after_rewrite)
        self.assertTrue("'NEW_TOKEN'" in rewritten)
        self.assertFalse("'TESTING'" in rewritten)
        self.assertFalse("'ANOT'" in rewritten)

    def test_include(self):
        t = loader.get_template('csrf_extended.html')
        c = Context({'csrf_token': 'TESTING', 'include_test': True})
        resp = t.render(c)
        parts = resp.rewritable_parts['csrf_token']
        before_rewrite = resp.replace(u"'ANOT'", '').replace(u"'TESTING'", '').replace(u"'ANOT2'", '')
        self.assertEquals(len(parts), 6)
        pos = 0
        for start, end in parts:
            pos += 1 
            if pos == 2:
                self.assertEquals(resp[start:end], u'ANOT')
            elif pos < 5:
                self.assertEquals(resp[start:end], u'TESTING')
            else:
                self.assertEquals(resp[start:end], u'ANOT2')
        rewritten = resp.rewrite({'csrf_token':'NEW_TOKEN'})
        after_rewrite = rewritten.replace("'NEW_TOKEN'", '')
        self.assertEquals(before_rewrite, after_rewrite)
        self.assertTrue("'NEW_TOKEN'" in rewritten)
        self.assertFalse("'TESTING'" in rewritten)
        self.assertFalse("'ANOT'" in rewritten)
        self.assertFalse("'ANOT2'" in rewritten)

    def test_rewritable_tag(self):
        t = loader.get_template('csrf_extended.html')
        c = Context({'csrf_token': 'TESTING', 'rewritable_test': True})
        resp = t.render(c)
        parts = resp.rewritable_parts['hello_user']
        self.assertEquals(len(parts), 1)
        for start, end in parts:
            self.assertTrue('Hello, SomeUser' in resp[start:end])
        # Currently, we must rewrite everything in the response
        rewritten = resp.rewrite({
            'csrf_token':'NEW_TOKEN',
            'hello_user': 'Hello, AnotherUser'
        })
        self.assertTrue('Hello, AnotherUser' in rewritten)
        self.assertFalse('Hello, SomeUser' in rewritten)
        self.assertTrue("'NEW_TOKEN'" in rewritten)
        self.assertFalse("'TESTING'" in rewritten)
