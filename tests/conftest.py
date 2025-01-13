import base64
import os
from unittest.mock import Mock
import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session
from src.data import SqlAlchemyBase


@pytest.fixture(scope="session")
def sample_html() -> str:
    return """
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


@pytest.fixture
def sample_html_data_url(sample_html) -> str:
    """Returns the sample HTML content as a base64 encoded data URI"""
    encoded_html = base64.b64encode(sample_html.encode("utf-8")).decode("utf-8")
    return f"data:text/html;base64,{encoded_html}"


@pytest.fixture
def mock_db():
    db = Mock()
    context = Mock()
    db.begin.return_value = context
    context.__enter__ = Mock()
    context.__exit__ = Mock()
    return db


@pytest.fixture
def sqlite_session():
    """Create an in-memory SQLite database session"""
    engine = create_engine("sqlite:///:memory:")
    SqlAlchemyBase.metadata.create_all(engine)

    with Session(engine) as session:
        yield session


@pytest.fixture
def tmp_db_url(tmp_path):
    """Create a temporary SQLite database file that gets cleaned up after use"""
    db_url = f"sqlite:///{tmp_path}/test.db"
    engine = create_engine(db_url)
    SqlAlchemyBase.metadata.create_all(engine)

    yield db_url

    engine.dispose()
