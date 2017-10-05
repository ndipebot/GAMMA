#ifndef MaterialManager_h
#define MaterialManager_h

class MaterialManager
{
  public:
  MaterialManager();
  virtual ~MaterialManager();
  virtual void execute() = 0;

  int updateMass_;
  
};

#endif
