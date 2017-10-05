#ifndef FluxManager_h
#define FluxManager_h

class FluxManager
{
  public:
  FluxManager();
  virtual ~FluxManager();
  virtual void execute() = 0;
};

#endif
