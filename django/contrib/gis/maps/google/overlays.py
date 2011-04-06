from django.utils.safestring import mark_safe
from django.contrib.gis.geos import fromstr, Point, LineString, LinearRing, Polygon

class GEvent(object):
    """
    A Python wrapper for wiring map events using the
    google.maps.event.addListener() function.

    Events can be attached to any object derived from GOverlayBase with the
    add_event() call.

    For more information please see the Google Maps API Reference:
     http://code.google.com/apis/maps/documentation/javascript/reference.html#event

    Example:

      from django.shortcuts import render_to_response
      from django.contrib.gis.maps.google import GoogleMap, GEvent, GPolyline

      def sample_request(request):
          polyline = GPolyline('LINESTRING(101 26, 112 26, 102 31)')
          event = GEvent('click',
            'function() { location.href = "http://www.google.com"}')
          polyline.add_event(event)
          return render_to_response('mytemplate.html',
          {'google' : GoogleMap(polylines=[polyline])})
    """

    def __init__(self, event, action):
        """
        Initializes a GEvent object.

        Parameters:

        event:
          string for the event, such as 'click'. The event must be a valid
          event for the object in the Google Maps API.
          There is no validation of the event type within Django.

        action:
          string containing a Javascript function, such as
          'function() { location.href = "newurl";}'
          The string must be a valid Javascript function. Again there is no
          validation fo the function within Django.
        """
        self.event = event
        self.action = action

    def __unicode__(self):
        "Returns the parameter part of a GEvent."
        return mark_safe('"%s", %s' %(self.event, self.action))

class GOverlayBase(object):

    JS_CLASSNAME = 'Overlay'

    def __init__(self):
        self.events = []

    def latlng_from_coords(self, coords):
        "Generates a JavaScript array of google.maps.LatLng objects for the given coordinates."
        return '[%s]' % ','.join(['new google.maps.LatLng(%s,%s)' % (y, x) for x, y in coords])

    def add_event(self, event):
        "Causes the event to be applied to the overlay object"
        self.events.append(event)

    def __unicode__(self):
        "The string representation is the JavaScript API call."
        return mark_safe('google.maps.%s(%s)' % (self.JS_CLASSNAME, self.js_params))

class GPolygon(GOverlayBase):
    """
    A Python wrapper for the google.maps.Polygon object.  For more information
    please see the Google Maps API Reference:
     http://code.google.com/apis/maps/documentation/javascript/reference.html#Polygon
    """

    JS_CLASSNAME = 'Polygon'

    def __init__(self, poly,
                 stroke_color='#0000ff', stroke_weight=2, stroke_opacity=1,
                 fill_color='#0000ff', fill_opacity=0.4):
        """
        The GPolygon object initializes on a GEOS Polygon or a
        parameter that may be instantiated into GEOS Polygon.  Please note
        that this will not depict a Polygon's internal rings.

        Keyword Options:

          stroke_color:
            The color of the polygon outline. Defaults to '#0000ff' (blue).

          stroke_weight:
            The width of the polygon outline, in pixels.  Defaults to 2.

          stroke_opacity:
            The opacity of the polygon outline, between 0 and 1.  Defaults to 1.

          fill_color:
            The color of the polygon fill.  Defaults to '#0000ff' (blue).

          fill_opacity:
            The opacity of the polygon fill.  Defaults to 0.4.
        """
        if isinstance(poly, basestring): poly = fromstr(poly)
        if isinstance(poly, (tuple, list)): poly = Polygon(poly)
        if not isinstance(poly, Polygon):
            raise TypeError('GPolygon may only initialize on GEOS Polygons.')

        # Getting the envelope of the input polygon (used for automatically
        # determining the zoom level).
        self.envelope = poly.envelope

        # Translating the coordinates into a JavaScript array of
        # Google `GLatLng` objects.
        self.points = self.latlng_from_coords(poly.shell.coords)

        # Stroke settings.
        self.stroke_color, self.stroke_opacity, self.stroke_weight = stroke_color, stroke_opacity, stroke_weight

        # Fill settings.
        self.fill_color, self.fill_opacity = fill_color, fill_opacity

        super(GPolygon, self).__init__()

    @property
    def js_params(self):
        result = []
        result.append('paths: %s' % self.points)
        if self.stroke_color: result.append('strokeColor: "%s"' % self.stroke_color)
        if self.stroke_weight: result.append('strokeWeight: "%s"' % self.stroke_weight)
        if self.stroke_opacity: result.append('strokeOpacity: "%s"' % self.stroke_opacity)
        if self.fill_color: result.append('fillColor: "%s"' % self.fill_color)
        if self.fill_opacity: result.append('fillOpactiy: "%s"' % self.fill_opacity)
        return '{%s}' % ','.join(result)

class GPolyline(GOverlayBase):
    """
    A Python wrapper for the google.maps.Polyline object.  For more information
    please see the Google Maps API Reference:
     http://code.google.com/apis/maps/documentation/javascript/reference.html#Polyline
    """

    JS_CLASSNAME = 'Polyline'

    def __init__(self, geom, color='#0000ff', weight=2, opacity=1):
        """
        The GPolyline object may be initialized on GEOS LineStirng, LinearRing,
        and Polygon objects (internal rings not supported) or a parameter that
        may instantiated into one of the above geometries.

        Keyword Options:

          color:
            The color to use for the polyline.  Defaults to '#0000ff' (blue).

          weight:
            The width of the polyline, in pixels.  Defaults to 2.

          opacity:
            The opacity of the polyline, between 0 and 1.  Defaults to 1.
        """
        # If a GEOS geometry isn't passed in, try to contsruct one.
        if isinstance(geom, basestring): geom = fromstr(geom)
        if isinstance(geom, (tuple, list)): geom = Polygon(geom)
        # Generating the lat/lng coordinate pairs.
        if isinstance(geom, (LineString, LinearRing)):
            self.latlngs = self.latlng_from_coords(geom.coords)
        elif isinstance(geom, Polygon):
            self.latlngs = self.latlng_from_coords(geom.shell.coords)
        else:
            raise TypeError('GPolyline may only initialize on GEOS LineString, LinearRing, and/or Polygon geometries.')

        # Getting the envelope for automatic zoom determination.
        self.envelope = geom.envelope
        self.color, self.weight, self.opacity = color, weight, opacity
        super(GPolyline, self).__init__()

    @property
    def js_params(self):
        result = []
        result.append('path: %s' % self.latlngs)
        if self.color: result.append('strokeColor: "%s"' % self.color)
        if self.weight: result.append('strokeWeight: "%s"' % self.weight)
        if self.opacity: result.append('strokeOpacity: "%s"' % self.opacity)
        return '{%s}' % ','.join(result)


class GImage(object):
    """
    Creates a GImage object to pass into a Gmarker object for the icon
    and shadow arguments.  The arguments are used to create a MarkerImage
    class in the javascript:

    http://code.google.com/apis/maps/documentation/javascript/reference.html#MarkerImage

    Required Arguments:

        url:
            The url of the image to be used as the icon on the map

    Keyword Options:

        size:
            a tuple representing the pixel size of the foreground (not the
            shadow) image of the icon, in the format: (width, height) ex.:

            GImage("/media/icon/star.png",
                  iconsize=(15,10))

            Would indicate your custom icon was 15px wide and 10px height.

        origin:
            a tuple representing the pixel coordinate of the upper left corner
            of the icon.  Used in conjuction with the size option to specify
            the sprite/subset of an image.  In the format: (x,y) ex.:

            3rd_marker = GImage("/media/icon/12_markers.png",
                               size=(15,10),
                               origin=(30,0))

            Would indicate the image where it's upper left corner is at (30,0)
            and its lower right corner is (45,10).

        anchor:
            a tuple representing the pixel coordinate relative to the top left
            corner of the icon image at which this icon is anchored to the map.
            In (x, y) format.  x increases to the right in the Google Maps
            coordinate system and y increases downwards in the Google Maps
            coordinate system.)

    """

    def __init__(self, url, size=None, origin=None, anchor=None):
        self.url = url
        self.size = size
        self.origin = origin
        self.anchor = anchor

    def _to_param(self):
        args = "(%s" % self.url
        if self.size:
            args += ", new google.maps.Size(%s)" % self.size
            if self.origin:
                args += ", new google.maps.Point(%s)" % self.origin
                if self.anchor:
                    args += ", new google.maps.Point(%s)" % self.anchor
        args += ")"
        return "new google.maps.MarkerImage(%s)" % args

class GMarker(GOverlayBase):
    """
    A Python wrapper for the Google GMarker object.  For more information
    please see the Google Maps API Reference:
     http://code.google.com/apis/maps/documentation/javascript/reference.html#Marker

    Example:

      from django.shortcuts import render_to_response
      from django.contrib.gis.maps.google.overlays import GMarker, GEvent

      def sample_request(request):
          marker = GMarker('POINT(101 26)')
          event = GEvent('click',
                         'function() { location.href = "http://www.google.com"}')
          marker.add_event(event)
          return render_to_response('mytemplate.html',
                 {'google' : GoogleMap(markers=[marker])})
    """

    JS_CLASSNAME = 'Marker'

    def __init__(self, geom, title=None, draggable=False, icon=None, shadow=None, visible=True):
        """
        The GMarker object may initialize on GEOS Points or a parameter
        that may be instantiated into a GEOS point.

        Keyword Options:
         title:
           Title option for GMarker, will be displayed as a tooltip.

         draggable:
           Draggable option for GMarker, disabled by default.

         icon:
           Sets the GImage used to display the marker on the map.
           If not set google maps will use the default marker icon.

         shadow:
           Sets the GImage used to display the shadow of the marker on the map.

         visible:
           Set if this marker will be visible at the map
        """
        # If a GEOS geometry isn't passed in, try to construct one.
        if isinstance(geom, basestring): geom = fromstr(geom)
        if isinstance(geom, (tuple, list)): geom = Point(geom)
        if isinstance(geom, Point):
            self.latlng = self.latlng_from_coords(geom.coords)
        else:
            raise TypeError('GMarker may only initialize on GEOS Point geometry.')
        # Getting the envelope for automatic zoom determination.
        self.envelope = geom.envelope
        # TODO: Add support for more GMarkerOptions
        self.title = title
        self.draggable = draggable
        self.icon = icon
        self.shadow = shadow
        self.visible = visible
        super(GMarker, self).__init__()

    def latlng_from_coords(self, coords):
        return 'new google.maps.LatLng(%s,%s)' %(coords[1], coords[0])

    @property
    def js_params(self):
        result = []
        result.append('position: %s' % self.latlng)
        if self.title: result.append('title: "%s"' % self.title)
        if self.icon: result.append('icon: %s' % self.icon._to_param())
        if self.shadow: result.append('shadow: %s' % self.shadow_to_param())
        if self.draggable: result.append('draggable: true')
        if not self.visible: result.append('visible: false')
        return '{%s}' % ','.join(result)
