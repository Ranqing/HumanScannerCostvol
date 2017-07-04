#ifndef HASHMAP_H
#define HASHMAP_H

#include "match.h"
#include "match_value.h"
#include "hash/hash.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;

typedef Mhash<Point2f, MatchValue, Hashfn1> MatchHashMap;

class MatchHash
{
public:
    MatchHash(const int cap_keys = 0, const int cap_values = 0);
    ~MatchHash();

    void init(const int cap_keys, const int cap_values) ;
    bool is_exist(const Point2f& key, const MatchValue& value) ;                        //判断匹配是否存在
    bool max_store(const Point2f& key, const MatchValue& value, const int maxcount );   //存储较好的匹配

    bool get_key(const int idx, Point2f& key);
    bool get_worst_value(const Point2f& key, MatchValue& worst_value);                  //get worst match corresponding to worst matching cost,
    bool get_value(const Point2f& key, MatchValue& value);                              //得到key对应的value

    int store(const Point2f& key, const MatchValue& value);
    int erase(const Point2f& key, const MatchValue& value);
    int get_count(const Point2f& key);
    int get_init(const Point2f& key);
    int get_next(MatchValue& value);

    void copy_in(const vector<Match>& matches);
    void copy_to(vector<Match>& matches);

    int get_num_of_keys()   { return m_num_keys; }
    int get_num_of_values() { return m_num_values;}

    void parse() ;
    bool db_parse(vector<float>& disp, vector<float>& prior, int w, int h, int len);

private:
    vector<Point2f> m_keys;       //keys
    MatchHashMap * m_hashmap;     //hash map

    int m_num_keys;               //num of keys
    int m_num_values;             //num of values
    int m_capcity;                //num of capcity
};

inline MatchHash::MatchHash(const int cap_keys, const int cap_values): m_hashmap(0), m_num_keys(0), m_num_values(0), m_capcity(cap_keys+cap_values) {

    if(cap_keys&&cap_values) {
        m_keys.clear();
        m_keys.resize(m_capcity);
        m_hashmap = new MatchHashMap(m_capcity, m_capcity);

        if(0==m_hashmap) {
            cerr << "failed to construct a hashmap..." << endl;
            exit(-1);
        }
    }
}

inline MatchHash::~MatchHash() {
    if(NULL != m_hashmap) {
        delete m_hashmap; m_hashmap = NULL;
    }
}


#endif // HASHMAP_H
