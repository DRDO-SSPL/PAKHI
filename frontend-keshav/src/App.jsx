import Notebook from './components/notebook';
import './index.css'; // Make sure this line is here to include styles

function App() {
  return (
    <div className="container">
      <h1 className="heading"> Notebook</h1>
      <Notebook />
    </div>
  );
}

export default App;
