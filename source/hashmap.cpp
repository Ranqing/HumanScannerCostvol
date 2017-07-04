#include "hashmap.h"

void MatchHash::init(const int cap_keys, const int cap_values) {
    if(NULL != m_hashmap) {
        delete m_hashmap; m_hashmap = NULL;
    }

    m_num_keys = 0;
    m_num_values = 0;
    m_capcity = cap_keys + cap_values;

    m_keys.clear();
    m_keys.resize(m_capcity);
    m_hashmap = new MatchHashMap(m_capcity, m_capcity);

    if(NULL == m_hashmap) {
        cerr << "failed to initial match hash map... " << endl;
        exit(-1);
    }
 }

bool MatchHash::is_exist(const Point2f &key, const MatchValue &value) {
    if(get_init(key)) {
        MatchValue t_value;
        while(get_next(t_value)) {
            if(t_value.m_d == value.m_d) return true;
        }
    }
    return false;
}


bool MatchHash::max_store(const Point2f &key, const MatchValue &value, const int maxcount) {
    int count = get_count(key);
    if(maxcount > count) {
        store(key, value);
        return true;
    }

    MatchValue worst_value;
    if(true == get_worst_value(key, worst_value)) {
        if(worst_value.m_mcost < value.m_mcost) {
            erase(key, worst_value);
            store(key, value);
            return true;
        }
    }
    return false;
}

bool MatchHash::get_key(const int idx, Point2f &key) {
    if(m_num_keys <= idx) {
        cerr << "out of num of keys..." << endl;
        return false;
    }
    key = m_keys[idx];
    return true;
}

//minimum
bool MatchHash::get_worst_value(const Point2f &key, MatchValue &worst_value) {
    MatchValue value;
    if(get_init(key)) {
        get_next(worst_value);
        while(get_next(value)) {
            if(value.m_mcost < worst_value.m_mcost) {
                worst_value = value;                              //select the minimum value of matching cost
            }
        }
    }
}

bool MatchHash::get_value(const Point2f &key, MatchValue &value) {
    if(get_init(key)) {
        get_next(value);
        return true;
    }
    return false;
}

int MatchHash::store(const Point2f &key, const MatchValue &value) {
    if(m_num_keys == m_capcity || m_num_values == m_capcity) {
        cerr << "match hash map is full while store()..." << endl;
        exit(-1);
    }

    int count = m_hashmap->count(key);
    if(0==count) {
        m_keys[m_num_keys++] = key;          //first time to store
    }
    m_num_values++;

    return m_hashmap->store(key, value);
}

int MatchHash::erase(const Point2f &key, const MatchValue &value) {
    int count = m_hashmap->count(key);
    if(0==count) {
        cerr << "no such pair in match hash map while erase()..." << endl;
        exit(-1);
    }
    if(1==count) {
        for(int i = 0; i < m_num_keys; ++i) {
            //  存储所有key的数组要删掉当前key, 并用最后一个补上；
            //  这样处理简单，但是会出现位置混乱
            if(m_keys[i] == key) {
                m_keys[i] = m_keys[m_num_keys - 1];
                m_keys[m_num_keys - 1] = m_keys[m_num_keys];
                m_num_keys --;
                break;
            }
        }
    }
    m_num_values --;
    return m_hashmap->erase(key, value);
}

int MatchHash::get_count(const Point2f &key) {
    return m_hashmap->count(key);
}

int MatchHash::get_init(const Point2f &key) {
    return m_hashmap->getinit(key);
}

int MatchHash::get_next(MatchValue &value) {
    return m_hashmap->getnext(value);
}

void MatchHash::copy_in(const vector<Match> &matches) {
    if(0==m_hashmap) {
        cerr << "fail to copy matches into match hashmap, hashmap should be initialized first.." << endl;
        exit(-1);
    }

    for(int i = 0, sz = matches.size(); i < sz; ++i) {
        Match tmatch = matches[i];

        Point2f key(tmatch.get_x(), tmatch.get_y());
        MatchValue value(tmatch.get_d(), tmatch.get_mcost(), tmatch.get_prior());

        store(key, value);
    }
# if 1
    cout << "copy matches vector into hashmap : num of matches = " << matches.size()
         << ". " << m_num_keys << " keys, " << m_num_values << " values." << endl;
#endif
}

void MatchHash::copy_to(vector<Match> &matches) {
    matches.clear(); matches.reserve(m_capcity);
    for(int i = 0, sz = m_num_keys; i < sz; ++i) {
        Point2f key = m_keys[i];
        MatchValue value;

        if(get_init(key)) {
            while(get_next(value)) {
                float x = key.x;
                float y = key.y;
                float d = value.m_d;
                float mcost = value.m_mcost;
                float prior = value.m_prior;

                matches.push_back(Match(x, y, d, mcost, prior));
            }
        }
    }
#if 1
    cout << "\tcopy hashmap out to matches vector : " << m_num_keys << " keys, " << m_num_values << " values. "
         << " num of matches = " << matches.size() << endl;
#endif

}

void MatchHash::parse() {
    for(int i = 0, sz = m_num_keys; i < sz; ++i) {
        Point2f t_key = m_keys[i];
        MatchValue t_value;
        if(get_init(t_key)) {
            while(get_next(t_value)) {
                cout << t_key << ", " << t_value ;
            }
        }
    }
}

//bool MatchHash::db_parse(vector<float>& disp, Mat &disp_3, float scale) {
//    int h = disp_3.size().height;
//    int w = disp_3.size().width;
//    int sx, sy, idx ;
//    float d;
//    bool ischanged = false;
//
//    unsigned char * ptr_disp_3 = (unsigned char *)disp_3.ptr<unsigned char>(0);
//
//    for(int i = 0, sz = m_num_keys; i < sz; ++i) {
//        Point2f t_key = m_keys[i];
//        sx = t_key.x;
//        sy = t_key.y;
//        idx = sy * w + sx;
//
//        MatchValue t_value;
//        if(get_init(t_key)) {
//            while(get_next(t_value)) {
//              //  cout << t_key << ", " << t_value ;
//                d = t_value.m_d;
//
//                if(d!=disp[idx]) {
//                    ischanged = true;
//                    disp[idx] = d;
//                    ptr_disp_3[3*idx+0] = disp[idx] * scale;
//                    ptr_disp_3[3*idx+2] = ptr_disp_3[3*idx+1] = ptr_disp_3[3*idx+0];
//                }
//
//            }
//        }
//    }
//    return ischanged;
//}

//db_parse(disp_vec, prior_vec, db_disp_3, db_prior_3, m_scale)

bool MatchHash::db_parse(vector<float>& disp, vector<float>& prior, int w, int h, int len) {

    int sx, sy, idx ;
    float d, p;
    bool ischanged = false;

    for(int i = 0, sz = m_num_keys; i < sz; ++i) {
        Point2f t_key = m_keys[i];
        sx = t_key.x;
        sy = t_key.y;
        idx = sy * w + sx;
        if(idx >= len) {
            cerr << "out of range in db_parse.." << endl;
            exit(1);
        }

        MatchValue t_value;
        if(get_init(t_key)) {
            while(get_next(t_value)) {
                d = t_value.m_d;
                p = t_value.m_prior;

                if(d!=disp[idx]) {
                    ischanged = true;
                    disp[idx] = d;
                    prior[idx] = p;
                }

            }
        }
    }
    return ischanged;

}

















