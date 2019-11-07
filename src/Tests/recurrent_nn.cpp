#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <algorithm>
#include <iterator>
#include <vector>
#include <algorithm>
#include <set>
#include <cctype>
#include <map>
#include <cfloat>
#include <cmath>
#include "blaze/Math.h"

using namespace std;



bool contains_weird_character(const std::string &c)
{
    if (c.find('0') != std::string::npos ||
        c.find('1') != std::string::npos ||
        c.find('2') != std::string::npos ||
        c.find('3') != std::string::npos ||
        c.find('4') != std::string::npos ||
        c.find('5') != std::string::npos ||
        c.find('6') != std::string::npos ||
        c.find('7') != std::string::npos ||
        c.find('8') != std::string::npos ||
        c.find('9') != std::string::npos)
    {
        return true;
    }

    return false;
}

std::string remove_weird_character(const std::string & str){

  std::string temp = str;
  temp.erase(std::remove(temp.begin(), temp.end(), '0'), temp.end());
  temp.erase(std::remove(temp.begin(), temp.end(), '1'), temp.end());
  temp.erase(std::remove(temp.begin(), temp.end(), '2'), temp.end());
  temp.erase(std::remove(temp.begin(), temp.end(), '3'), temp.end());
  temp.erase(std::remove(temp.begin(), temp.end(), '4'), temp.end());
  temp.erase(std::remove(temp.begin(), temp.end(), '5'), temp.end());
  temp.erase(std::remove(temp.begin(), temp.end(), '6'), temp.end());
  temp.erase(std::remove(temp.begin(), temp.end(), '7'), temp.end());
  temp.erase(std::remove(temp.begin(), temp.end(), '8'), temp.end());
  temp.erase(std::remove(temp.begin(), temp.end(), '9'), temp.end());
  temp.erase(std::remove(temp.begin(), temp.end(), '?'), temp.end());
  temp.erase(std::remove(temp.begin(), temp.end(), '.'), temp.end());
  return temp;
}

std::vector<double> softmax(const std::vector<double> & x){
  std::vector<double> soft(x.size());
  double sum = 0.;
  double max = DBL_MIN;
  for (int i = 0; i < x.size(); i++){
    max = (x[i] > max) ? x[i] : max;
  }  
  for (int i = 0; i < x.size(); i++){
    soft[i] = exp(x[i] - max);
    sum += soft[i];
  }
  for (int i = 0; i < x.size(); i++){
    soft[i] /= sum;
  }
  return soft;
}

std::vector<int> words2indices(std::vector<string> sentence,
			       std::map<string,int> word2index){
  std::vector<int> idx;
  for (auto i : sentence){
    idx.push_back(word2index[i]);
  }
  return idx;
}

std::vector<std::string> split(const std::string &text, char sep) {
  std::vector<std::string> tokens;
  std::size_t start = 0, end = 0;
  while ((end = text.find(sep, start)) != std::string::npos) {
    tokens.push_back(text.substr(start, end - start));
    start = end + 1;
  }
  tokens.push_back(text.substr(start));
  return tokens;
}


struct Layer
{
  blaze::DynamicVector<double,blaze::rowVector> pred;
  blaze::DynamicVector<double,blaze::rowVector> hidden;
  blaze::DynamicVector<double,blaze::rowVector> output_delta;
  blaze::DynamicVector<double,blaze::rowVector> hidden_delta;
  Layer(int embed_size, int vocab_size) {
    pred.resize(vocab_size);
    hidden.resize(embed_size);
  }
};


std::pair<std::vector<Layer>, double> forward_prop
(
 const std::vector<int> & sent,
 const blaze::DynamicVector<double,blaze::rowVector> & start,
 const blaze::DynamicMatrix<double,blaze::rowVector> & recurrent,
 const blaze::DynamicMatrix<double,blaze::rowMajor> & embed,
 const blaze::DynamicMatrix<double,blaze::rowMajor> & decoder,
 int embed_size,
 int vocab_size
 )
{
  std::vector<Layer> layers;
  Layer layer(embed_size, vocab_size);
  layer.hidden = start;
  layers.push_back(layer);
  double loss = 0;

  for (int target_i = 0; target_i < sent.size(); target_i++){
    Layer layer(embed_size, vocab_size);
    layer.pred = softmax(layers[layers.size()-1].hidden*decoder);
    loss += log(layer.pred[sent[target_i]]);
    layer.hidden = layers[layers.size()-1].hidden*recurrent + row(embed,sent[target_i]);
    layers.push_back(layer);
  }

  return std::pair<std::vector<Layer>,double>(layers, loss);
}


void back_prop(
	       const blaze::DynamicMatrix<double,blaze::rowVector> & recurrent,
	       const blaze::DynamicMatrix<double,blaze::rowMajor> & decoder,
	       const std::vector<int> & sent,
	       std::vector<Layer> & layers,
	       blaze::DynamicMatrix<double,blaze::rowVector> one_hot
	      )
{

  blaze::DynamicMatrix<double,blaze::rowVector> decoder_transpose = decoder;
  decoder_transpose.transpose();
  for (int layer_idx = layers.size()-1; layer_idx >= 0; layer_idx--){
    Layer layer = layers[layer_idx];
      // target = sent[layer_idx-1];
    int target = sent[layer_idx-1];
      
    if (layer_idx > 0){
      layer.output_delta = layer.pred - row(one_hot,target);

      
      
      blaze::DynamicVector<double,blaze::rowVector> new_hidden_delta = layer.output_delta*decoder_transpose;

      if (layer_idx == layers.size()-1){
	layer.hidden_delta = new_hidden_delta;
      }
      else {
	auto recurrent_transpose = recurrent;
	recurrent_transpose.transpose();
	
	layer.hidden_delta = new_hidden_delta +
  layers[layer_idx+1].hidden_delta*recurrent_transpose;
  	}
      }
      else {
	auto recurrent_transpose = recurrent;
	recurrent_transpose.transpose();
	
  	layer.hidden_delta =
  layers[layer_idx+1].hidden_delta*recurrent_transpose;
      }
    }
}

void weight_update
(
 double alpha,
 const std::vector<int> & sent,
 const std::vector<Layer> & layers,
 blaze::DynamicVector<double,blaze::rowVector> & start,
 blaze::DynamicMatrix<double,blaze::rowVector> & recurrent,
 blaze::DynamicMatrix<double,blaze::rowMajor> & embed,
 blaze::DynamicMatrix<double,blaze::rowMajor> & decoder
 )
{
  start -= layers[0].hidden_delta * alpha / (double)sent.size();
  for (int layer_idx = 1; layer_idx < layers.size(); layer_idx++){
      
    const Layer& layer = layers[layer_idx];

    blaze::DynamicVector<double,blaze::columnVector> output_delta_transpose(layer.output_delta.size());

    for (int i = 0; i < layer.output_delta.size(); i++){
      output_delta_transpose[i] = layer.output_delta[i];
    }
    
    blaze::DynamicVector<double,blaze::columnVector> hidden_delta_transpose(layer.hidden_delta.size());

    for (int i = 0; i < layer.hidden_delta.size(); i++){
      hidden_delta_transpose[i] = layer.hidden_delta[i];
    }
    
    decoder -= layer.hidden * output_delta_transpose * alpha / (double)sent.size();
    int embed_idx = sent[layer_idx];
    row(embed,embed_idx) -= layer.hidden_delta * alpha / (double)sent.size();
    recurrent -= layer.hidden*hidden_delta_transpose * alpha / (double)sent.size();
      
  }      
}


template<typename T, typename P>
T remove_if(T beg, T end, P pred)
{
    T dest = beg;
    for (T itr = beg;itr != end; ++itr)
        if (!pred(*itr))
            *(dest++) = *itr;
    return dest;
}


int main () {

  std::string line;
  std::ifstream myfile ("qa1_single-supporting-fact_train.txt");

  std::vector<std::vector<std::string>> tokens;
  int linenum = 0;
  if (myfile.is_open()){
      while(getline(myfile,line)){
	if (linenum > 1000){
	  break;
	}
        std::vector<string> tokens_tmp = split(line, ' ');
	line.erase(std::remove_if(line.begin(), line.end(), ::isspace), line.end());
        tokens_tmp.erase(tokens_tmp.begin());
        tokens.push_back(tokens_tmp);
	linenum++;
      }
      myfile.close();
  }

  std::set<string> words_set;
  for (int i = 0; i < tokens.size(); ++i) {
    for (int j = 0; j < tokens[i].size(); j++){
      // if(!contains_weird_character(tokens[i][j])){
	// words_set.emplace(remove_weird_character(tokens[i][j]));
	words_set.emplace(tokens[i][j]);
      // }
    }
  }

  std::cout << words_set.size() << std::endl;

  std::vector<string> vocab =  {"mary", "\tkitchen\t2", "\tgarden\t7", "\tkitchen\t7", "\tkitchen\t13", "john", "\tgarden\t1", "\tkitchen\t1", "\tgarden\t10", "bathroom.", "\tbedroom\t5", "\toffice\t4", "\tgarden\t4", "\tbedroom\t2", "where", "kitchen.", "\thallway\t13", "office.", "\tbedroom\t7", "\tbathroom\t1", "the", "\tkitchen\t10", "\thallway\t2", "\tgarden\t8", "\tbedroom\t4", "\toffice\t11", "\thallway\t8", "\thallway\t10", "\tkitchen\t11", "\tkitchen\t5", "to", "\tkitchen\t14", "mary?", "\toffice\t1", "\tbedroom\t1", "\thallway\t5", "garden.", "\tbedroom\t8", "\tgarden\t11", "\thallway\t11", "\tbedroom\t11", "\tgarden\t14", "\tbathroom\t14", "\thallway\t7", "is", "\tbedroom\t10", "moved", "\tbathroom\t8", "daniel", "\thallway\t1", "\tbedroom\t13", "sandra", "travelled", "\tbathroom\t11", "\tbedroom\t14", "went", "\tbathroom\t7", "\toffice\t2", "back", "\toffice\t8", "\toffice\t10", "daniel?", "\tgarden\t13", "\tgarden\t5", "john?", "journeyed", "\toffice\t13", "\tgarden\t2", "\tbathroom\t10", "\tkitchen\t4", "bedroom.", "hallway.", "\thallway\t14", "\tbathroom\t2", "\tbathroom\t5", "\tkitchen\t8", "sandra?", "\thallway\t4", "\toffice\t5", "\toffice\t14", "\tbathroom\t13", "\tbathroom\t4"};
  
  std::map<string,int> words2ind;
  for (int i = 0; i < vocab.size(); i++){
    words2ind.insert(std::pair<string,int>(vocab[i],i));
  }

  int embed_size = 10;
  double alpha = .001;
  //word embeddings
  blaze::DynamicMatrix<double,blaze::rowMajor> embed( vocab.size(), embed_size );

  //embedding -> output weights
  blaze::DynamicMatrix<double,blaze::rowMajor> decoder( embed_size, vocab.size() );

  // embedding -> embedding
  blaze::DynamicMatrix<double,blaze::rowVector> recurrent( embed_size, embed_size, 1.0);

  // one hot lookups (for loss function)
  blaze::DynamicMatrix<double,blaze::rowMajor> one_hot( vocab.size(), vocab.size(), 0.0);

  for (int i = 0; i < vocab.size(); i++)
      one_hot(i,i) = 1.;
  
  //sentence embedding for empty sentence
  blaze::DynamicVector<double,blaze::rowVector> start( embed_size, 0.0);
  
  for (int i = 0; i < vocab.size(); i++){
    for (int j = 0; j < embed_size; j++) {
      embed(i, j) = 0.1;//(0.1*((double)rand()/(double)RAND_MAX) - 0.5);
      decoder(j, i) = 0.1;//(0.1*((double)rand()/(double)RAND_MAX) - 0.5);
    }
  }

  std::vector<int> sent = words2indices(tokens[0],words2ind);
  std::pair<std::vector<Layer>, double> result = forward_prop(sent,start,recurrent,embed,decoder,
							      embed_size, vocab.size());

  
  for (int i = 0; i < 30000; i++){

    std::vector<int> sent = words2indices(tokens[i % tokens.size()],words2ind);
    std::pair<std::vector<Layer>,double> forward_result = forward_prop(sent,start,recurrent,embed,decoder,embed_size, vocab.size());
    std::vector<Layer> & layers = forward_result.first;
    double & loss = forward_result.second;
    back_prop(recurrent,decoder,sent,layers,one_hot);

    weight_update(alpha,sent,layers,start,recurrent,embed,decoder);
 
    if (i % 1000 == 0) {
      std::cout << "Perplexity:" << exp(loss / sent.size()) << std::endl;
    }
  }
    
  
}
