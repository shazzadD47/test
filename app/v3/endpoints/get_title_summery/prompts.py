PROMPT_LEGEND_DETECTION = """
You are given two images.The first image is the overall figure and
the second image is a zoomed in subfigure
Does the second image have the required legend area
(both legend names and markers) present in the first image to identify its lines.
Marker information must be present in the legend area for the second image
Legend names present with the lines is also acceptable
write yes or no

                            """

PROMPT_AREA_COMPARE = """
You are given two images.The first image is the overall figure and
the second image is a zoomed in subfigure
which green area  in the first image can be used to map the both
legend names and markers in the second image. Carefully
choose the green area which can be used to map the legends in the second image.
The area chosen must have legend and marker symbols. Do not choose a non-legend area.
Choose the most proper green area that can be used to map the legends in the
second image.
Does the second image need any legend area (with markers) to be added from
the first image  write yes or no.
just give the number of the green area in your output

example
"area_number_to_choose":"0"

"""
