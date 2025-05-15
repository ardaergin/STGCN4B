from rdflib import Graph, Namespace, RDFS, XSD, OWL

class NamespaceMixin:
    """Standard RDF namespaces used across the OfficeGraph & floor-plan modules."""

    IC        = Namespace("https://interconnectproject.eu/example/")

    SAREF     = Namespace("https://saref.etsi.org/core/")
    S4ENER    = Namespace("https://saref.etsi.org/saref4ener/")
    S4BLDG    = Namespace("https://saref.etsi.org/saref4bldg/")

    BOT       = Namespace("https://w3id.org/bot#")
    OM        = Namespace("http://www.wurvoc.org/vocabularies/om-1.8/")

    GEO       = Namespace("http://www.w3.org/2003/01/geo/wgs84_pos#")
    GEOSPARQL = Namespace("http://www.opengis.net/ont/geosparql#")

    EX        = Namespace("https://example.org/")
    EX_ONT    = Namespace("https://example.org/ontology#")

    RDFS      = RDFS
    XSD       = XSD
    OWL       = OWL

    @classmethod
    def create_empty_graph_with_namespace_bindings(cls) -> Graph:
        """
        Create a new rdflib Graph with all CoreNamespaces (plus rdfs & xsd) bound.
        Prefixes are taken as the lowercase of each attribute name.
        """
        graph = Graph()

        # Explicitly binding each namespace
        graph.bind("ic", cls.IC)

        graph.bind("saref", cls.SAREF)
        graph.bind("s4ener", cls.S4ENER)
        graph.bind("s4bldg", cls.S4BLDG)

        graph.bind("bot", cls.BOT)
        graph.bind("om", cls.OM)

        graph.bind("geo", cls.GEO)
        graph.bind("geosparql", cls.GEOSPARQL)

        graph.bind("ex", cls.EX)
        graph.bind("ex-ont", cls.EX_ONT)
        
        # Standard prefixes
        graph.bind("rdfs", cls.RDFS)
        graph.bind("xsd", cls.XSD)
        graph.bind("owl", cls.OWL)

        return graph
