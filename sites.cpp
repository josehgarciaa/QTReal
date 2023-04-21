#include "sites.h"

Site::Site(Position X, size_t tag) : position(X), tag(tag){
};

Site::Site():position(Position()){};

Site::~Site(){
    position = std::move(Position());
    tag = 0;
    binIdx = Index::Zero();
}


Site::Site(const Site& A){
    this -> tag = A.tag;
    this -> binIdx = A.binIdx;
    this -> position = A.position;
}

Site::Site(Site&& A){
    this -> tag = std::move(A.tag);
    this -> binIdx = std::move(A.binIdx);
    this -> position = std::move(A.position);
 }


Site & Site::operator = (const Site& A){
    if(&A == this)
        return *this;
    else{
        this -> tag = A.tag;
        this -> binIdx = A.binIdx;
        this -> position = A.position;
        return *this;
    }
}


Site & Site::operator=(Site &&A ){
    if(this !=&A){
        this -> tag = std::move(A.tag);
        this -> binIdx = std::move(A.binIdx);
        this -> position = std::move(A.position);
        return *this;
    }
    else{
        return *this;
    }
}


Float Site::distance(Site & A){
    Position dx = A.position;
    dx -= this -> position;
    return dx.norm();
}

Position Site::displacement(Site & A){
    Position dx = A.position;
    return dx-= this-> position;
}

