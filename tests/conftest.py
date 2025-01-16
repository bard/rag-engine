from langchain.schema import Document
import base64
import pytest
from unittest.mock import Mock
from sqlalchemy import create_engine
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
def sample_html_as_data_url(sample_html) -> str:
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


@pytest.fixture(scope="module")
def vcr_config():
    return {"filter_headers": ["authorization"]}


@pytest.fixture
def insurance_data_documents():
    return [
        Document(page_content=content, id=str(i))
        for i, content in enumerate(
            [
                "Year: 2012, Expenditure: $812.40, Change: 2.2%",
                "Year: 2013, Expenditure: $841.06, Change: 3.5%",
                "Year: 2014, Expenditure: $869.47, Change: 3.4%",
                "Year: 2015, Expenditure: $896.66, Change: 3.1%",
                "Year: 2016, Expenditure: $945.22, Change: 5.4%",
                "Year: 2017, Expenditure: $1008.35, Change: 6.7%",
                "Year: 2018, Expenditure: $1058.10, Change: 4.9%",
                "Year: 2019, Expenditure: $1071.74, Change: 1.3%",
                "Year: 2020, Expenditure: $1046.37, Change: -2.4%",
                "Year: 2021, Expenditure: $1061.54, Change: 1.4%",
            ]
        )
    ]


@pytest.fixture
def travel_info_documents():
    return [
        Document(page_content=content, id=str(i))
        for i, content in enumerate(
            [
                "Paris, the 'City of Light,' boasts iconic landmarks such as the Eiffel Tower, offering panoramic views from its observation decks. The Louvre Museum, home to masterpieces like the Mona Lisa, attracts art enthusiasts worldwide. Notre-Dame Cathedral, a Gothic architectural marvel, stands on the Île de la Cité. The Champs-Élysées, lined with shops and cafes, leads to the Arc de Triomphe, honoring those who fought for France. Montmartre, with its artistic heritage, features the Basilica of the Sacré-Cœur atop its hill.",
                "London, a vibrant metropolis, features the historic Tower of London, housing the Crown Jewels. The British Museum showcases a vast collection of world art and artifacts. Buckingham Palace, the monarch's residence, is famed for the Changing of the Guard ceremony. The Houses of Parliament and Big Ben are iconic symbols along the River Thames. The London Eye offers a rotating perspective of the city's skyline.",
                "Rome, the 'Eternal City,' is home to the Colosseum, an ancient amphitheater echoing gladiatorial history. The Vatican City enclaves St. Peter's Basilica and the Sistine Chapel, adorned with Michelangelo's frescoes. The Pantheon, with its impressive dome, stands as a testament to Roman engineering. The Roman Forum's ruins narrate tales of imperial grandeur. Trevi Fountain invites visitors to toss a coin, ensuring their return to Rome.",
                "Berlin, Germany's capital, presents the Brandenburg Gate, a neoclassical monument symbolizing unity. The Berlin Wall Memorial commemorates the city's divided past. Museum Island hosts a cluster of renowned museums, including the Pergamon Museum. The Reichstag Building, with its modern glass dome, offers insights into German politics. Checkpoint Charlie stands as a relic of Cold War history.",
                "Madrid, Spain's lively capital, features the Prado Museum, exhibiting European art masterpieces. The Royal Palace impresses with its opulent architecture and historic significance. Retiro Park provides a green oasis in the city's heart. Puerta del Sol serves as a bustling public square and focal point for festivities. The Gran Vía is renowned for shopping, theaters, and vibrant nightlife.",
            ]
        )
    ]
