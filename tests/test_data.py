import base64
from src.data import parse_insurance_table, load_insurance_data


SAMPLE_HTML = """
<html>
  <head>
    <meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
  </head>
  <body>
    <table>
    <thead><tr><th></th></tr></thead>    <tbody> <tr class="odd"><td><table>
<tr><td>Average Expenditures for Auto Insurance, 2012-2021</td></tr>
<tr><td><p> </p>
</td></tr>
<tr><td><table>
<thead>
<tr>
<th>Year</th>
<th>Average expenditure</th>
<th>Percent change</th>
</tr>
</thead>
<tbody>
<tr>
<td>2012</td>
<td>$812.40</td>
<td>2.2%</td>
</tr>
<tr>
<td>2013</td>
<td>841.06</td>
<td>3.5</td>
</tr>
<tr>
<td>2014</td>
<td>869.47</td>
<td>3.4</td>
</tr>
<tr>
<td>2015</td>
<td>896.66</td>
<td>3.1</td>
</tr>
<tr>
<td>2016</td>
<td>945.22</td>
<td>5.4</td>
</tr>
<tr>
<td>2017</td>
<td>1,008.35</td>
<td>6.7</td>
</tr>
<tr>
<td>2018</td>
<td>1,058.10</td>
<td>4.9</td>
</tr>
<tr>
<td>2019</td>
<td>1,071.74</td>
<td>1.3</td>
</tr>
<tr>
<td>2020</td>
<td>1,046.37</td>
<td>-2.4</td>
</tr>
<tr>
<td>2021</td>
<td>1,061.54</td>
<td>1.4</td>
</tr>
</tbody></table>

Source: National Association of Insurance Commissioners (NAIC). Further reprint or distribution strictly prohibited without written permission of NAIC.</td></tr>
</table></td> </tr>
      </tbody>
    </table>
  </body>
</html>
"""


def test_parse_insurance_table(snapshot):
    result = parse_insurance_table(SAMPLE_HTML)
    assert result == snapshot


def test_load_insurance_data_with_data_url(snapshot):
    html_bytes = SAMPLE_HTML.encode("utf-8")
    encoded_html = base64.b64encode(html_bytes).decode("utf-8")

    data_url = f"data:text/html;base64,{encoded_html}"
    documents = load_insurance_data(data_url)
    assert (len(documents)) == 10
    assert documents[0] == snapshot
